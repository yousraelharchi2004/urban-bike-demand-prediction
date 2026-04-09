"""
Export Blueprint - Data export and email routes.
Routes: /export/*, /api/export/*
"""

from flask import Blueprint, request, jsonify, make_response
import io
import json
import os
import zipfile
from datetime import datetime

from app.blueprints.analytics import _loaded_models, get_model_meta
from app.email_service import email_service
from app.model_metrics import calculate_model_metrics, generate_performance_report
from app.models.loader import get_state
from app.services.export_service import filtered_history, history_rows, predictions_to_csv, predictions_to_json

export = Blueprint('export', __name__)


def _email_config():
    return {
        'smtp_server': os.getenv('SMTP_SERVER', email_service.smtp_server),
        'smtp_port': int(os.getenv('SMTP_PORT', str(email_service.smtp_port))),
        'sender_email': os.getenv('SENDER_EMAIL', email_service.sender_email),
        'sender_password': os.getenv('SENDER_PASSWORD', email_service.sender_password),
        'from_email': os.getenv('EMAIL_FROM', os.getenv('SENDER_EMAIL', email_service.sender_email or 'noreply@urbanflow.com')),
    }


def _chart_payload(chart_type, time_filter='all'):
    state = get_state()

    if chart_type == 'feature_importance':
        search_order = ['xgboost', 'stacking', 'lightgbm', 'advanced', 'improved', 'original']
        model = None
        source = 'original'
        for key in search_order:
            candidate = state.get(f'{key}_model')
            if candidate and hasattr(candidate, 'feature_importances_') and hasattr(candidate, 'feature_names_in_'):
                model = candidate
                source = key
                break
        if model is None:
            return None
        pairs = sorted(
            zip(list(model.feature_names_in_), model.feature_importances_.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )[:12]
        return {
            'type': 'bar',
            'data': {
                'labels': [name for name, _ in pairs],
                'datasets': [{
                    'label': f"Feature Importance ({source})",
                    'data': [round(value * 100, 2) for _, value in pairs],
                    'backgroundColor': '#3b82f6',
                    'borderColor': '#1d4ed8',
                }],
            },
            'options': {'title': 'Feature Importance', 'ylabel': 'Importance %'},
        }

    if chart_type == 'hourly_patterns':
        history = filtered_history(time_filter)
        hourly = {hour: [] for hour in range(24)}
        for item in history:
            inp = item.get('input_data', {})
            hour = inp.get('Hour', item.get('hour'))
            value = item.get('predicted_bikes', 0)
            if hour is not None and value is not None:
                hourly[int(hour)].append(float(value))
        return {
            'type': 'line',
            'data': {
                'labels': [f'{hour:02d}:00' for hour in range(24)],
                'datasets': [{
                    'label': 'Average Demand',
                    'data': [round(sum(values) / len(values), 1) if values else 0 for _, values in hourly.items()],
                    'backgroundColor': '#0d9488',
                    'borderColor': '#0f766e',
                }],
            },
            'options': {'title': 'Hourly Demand Patterns', 'ylabel': 'Predicted Bikes'},
        }

    if chart_type == 'performance_radar':
        loaded = _loaded_models(state)
        model_meta = get_model_meta()
        palette = {
            'original': '#3b82f6',
            'improved': '#6366f1',
            'advanced': '#0d9488',
            'xgboost': '#8b5cf6',
            'lightgbm': '#06b6d4',
            'stacking': '#64748b',
            'unified': '#0f766e',
        }
        datasets = []
        for key in loaded:
            meta = model_meta.get(key)
            if not meta:
                continue
            datasets.append({
                'label': meta['name'],
                'data': [meta['accuracy'], meta['stability'], meta['speed'], meta['r2'] * 100, max(0, 100 - meta['mae'] / 3)],
                'borderColor': palette.get(key, '#3b82f6'),
                'backgroundColor': palette.get(key, '#3b82f6'),
            })
        return {
            'type': 'radar',
            'data': {
                'labels': ['Accuracy', 'Stability', 'Speed', 'R2 Score', 'Low Error'],
                'datasets': datasets,
            },
            'options': {'title': 'Model Performance Radar'},
        }

    return None


def _generate_chart_image(chart_data, format_type='png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 6), dpi=100)
    FigureCanvas(fig)
    labels = chart_data['data']['labels']
    datasets = chart_data['data']['datasets']
    chart_kind = chart_data['type']

    if chart_kind == 'radar':
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        for dataset in datasets:
            values = list(dataset['data']) + [dataset['data'][0]]
            color = dataset.get('borderColor', '#3b82f6')
            ax.plot(angles, values, linewidth=2, label=dataset['label'], color=color)
            ax.fill(angles, values, alpha=0.2, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    else:
        ax = fig.add_subplot(111)
        x = np.arange(len(labels))
        data = datasets[0]['data']
        background = datasets[0].get('backgroundColor', '#3b82f6')
        border = datasets[0].get('borderColor', '#1d4ed8')
        if chart_kind == 'bar':
            ax.bar(x, data, color=background, edgecolor=border)
        else:
            ax.plot(x, data, color=border, linewidth=2.5, marker='o')
            ax.fill_between(x, data, alpha=0.2, color=background)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(chart_data.get('options', {}).get('ylabel', 'Value'))

    ax.set_title(chart_data.get('options', {}).get('title', 'UrbanFlow Chart'))
    fig.tight_layout()

    buffer = io.BytesIO()
    if format_type == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(buffer) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
    else:
        fig.savefig(buffer, format=format_type, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return buffer.getvalue()


def _metrics_pdf_response(metrics):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("UrbanFlow Model Metrics Report", styles['Title']),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
        Spacer(1, 20),
    ]

    data = [
        ['Metric', 'Value'],
        ['R2 Score', f"{metrics.get('r2_score', 0):.3f}"],
        ['Mean Absolute Error', f"{metrics.get('mean_absolute_error', 0):.1f} bikes"],
        ['Root Mean Squared Error', f"{metrics.get('root_mean_squared_error', 0):.1f} bikes"],
        ['Features Count', str(metrics.get('features_count', 0))],
        ['Model Name', metrics.get('model_name', 'Unknown')],
    ]

    table = Table(data, colWidths=[2.8 * inch, 3.4 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(table)
    doc.build(story)

    response = make_response(buffer.getvalue())
    response.headers["Content-Disposition"] = f'attachment; filename="urbanflow_metrics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
    response.headers["Content-Type"] = "application/pdf"
    return response


@export.route('/export/predictions/csv', methods=['GET'])
def export_predictions_csv():
    """Export prediction history as CSV."""
    try:
        rows = history_rows(filtered_history('all'))
        response = make_response(predictions_to_csv(rows))
        response.headers["Content-Disposition"] = "attachment; filename=predictions_history.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/export/predictions/json', methods=['GET'])
def export_predictions_json():
    """Export prediction history as JSON."""
    try:
        rows = history_rows(filtered_history('all'))
        response = make_response(predictions_to_json(rows))
        response.headers["Content-Disposition"] = "attachment; filename=predictions_history.json"
        response.headers["Content-Type"] = "application/json"
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/export/model-report', methods=['GET'])
def export_model_report():
    """Export a model performance report as text."""
    try:
        report = generate_performance_report()
        response = make_response(report)
        response.headers["Content-Disposition"] = "attachment; filename=model_report.txt"
        response.headers["Content-Type"] = "text/plain"
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/export/metrics/pdf', methods=['GET'])
def export_metrics_pdf():
    """Export model metrics as PDF."""
    try:
        return _metrics_pdf_response(calculate_model_metrics())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/api/send-analytics-email', methods=['POST'])
def send_analytics_email():
    """Send analytics email via the modular email service."""
    try:
        email_data = request.get_json() or {}
        config = _email_config()
        if not config['sender_email'] or not config['sender_password']:
            return jsonify({
                'success': False,
                'error': 'Email service not configured. Please set SENDER_EMAIL and SENDER_PASSWORD.',
                'config_required': True,
            }), 503

        result = email_service.send_analytics_report(email_data)
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': result.get('message', 'Email sent successfully'),
                'attachments_count': result.get('attachments_count', 0),
            })
        return jsonify({'success': False, 'error': result.get('error', 'Email sending failed')}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@export.route('/export/email', methods=['POST'])
def export_via_email():
    """Compatibility endpoint for simple email export requests."""
    try:
        data = request.get_json() or {}
        email_to = data.get('email')
        if not email_to:
            return jsonify({"success": False, "error": "Email address is required"}), 400

        config = _email_config()
        if not config['sender_email'] or not config['sender_password']:
            return jsonify({"success": False, "error": "Email service not configured"}), 503

        email_data = {
            'recipients': [email_to],
            'subject': f"UrbanFlow {data.get('type', 'report').title()} Report - {datetime.now().strftime('%Y-%m-%d')}",
            'message': data.get('message') or 'Please find attached the UrbanFlow report.',
            'includePDF': data.get('include_charts', True),
            'includeCSV': data.get('include_csv', True),
            'includeCharts': data.get('include_charts', True),
            'schedule': 'once',
        }

        result = email_service.send_analytics_report(email_data)
        if result.get('success'):
            return jsonify({
                "success": True,
                "message": f"Report sent successfully to {email_to}",
                "email": email_to,
                "type": data.get('type', 'report'),
                "timestamp": datetime.now().isoformat(),
            })
        return jsonify({"success": False, "error": result.get('error', 'Failed to send email')}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/export/email/test', methods=['GET'])
@export.route('/api/test-email', methods=['GET'])
def test_email_config():
    """Compatibility email configuration check."""
    try:
        config = _email_config()
        if not config['sender_email'] or not config['sender_password']:
            return jsonify({
                "success": False,
                "error": "Email service not configured",
                "email_config": {
                    "host": config['smtp_server'],
                    "port": config['smtp_port'],
                    "sender_email_set": bool(config['sender_email']),
                    "from_email": config['from_email'],
                },
            }), 503

        result = email_service.send_analytics_report({
            'recipients': [config['sender_email']],
            'subject': 'UrbanFlow Email Test',
            'message': 'This is a test email from UrbanFlow.',
            'includePDF': False,
            'includeCSV': False,
            'includeCharts': False,
            'schedule': 'once',
        })
        if result.get('success'):
            return jsonify({
                "success": True,
                "message": "Email test completed successfully",
                "email_config": {
                    "host": config['smtp_server'],
                    "port": config['smtp_port'],
                    "sender_email_set": bool(config['sender_email']),
                    "from_email": config['from_email'],
                },
            })
        return jsonify({"success": False, "error": result.get('error', 'Email test failed')}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/api/export/chart/<chart_type>', methods=['GET'])
def export_chart(chart_type):
    """Export a generated analytics chart."""
    try:
        format_type = request.args.get('format', 'png').lower()
        time_filter = request.args.get('filter', 'all')
        if format_type not in ('png', 'pdf', 'svg', 'jpg'):
            return jsonify({"success": False, "error": "Invalid format"}), 400

        chart_data = _chart_payload(chart_type, time_filter)
        if not chart_data:
            return jsonify({"success": False, "error": "Invalid or unavailable chart type"}), 400

        image_format = 'jpeg' if format_type == 'jpg' else format_type
        binary = _generate_chart_image(chart_data, image_format)
        filename = f'urbanflow_{chart_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format_type}'
        response = make_response(binary)
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['Content-Type'] = 'application/pdf' if format_type == 'pdf' else f'image/{image_format}'
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/api/export/all-charts', methods=['GET'])
def export_all_charts():
    """Export all server-generated charts as a zip archive."""
    try:
        format_type = request.args.get('format', 'png').lower()
        time_filter = request.args.get('filter', 'all')
        if format_type not in ('png', 'pdf'):
            return jsonify({"success": False, "error": "Invalid format"}), 400

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for chart_type in ('feature_importance', 'hourly_patterns', 'performance_radar'):
                chart_data = _chart_payload(chart_type, time_filter)
                if not chart_data:
                    continue
                archive.writestr(
                    f'urbanflow_{chart_type}.{format_type}',
                    _generate_chart_image(chart_data, format_type),
                )
            archive.writestr(
                'README.txt',
                f'UrbanFlow chart export\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\nFilter: {time_filter}\n',
            )

        response = make_response(zip_buffer.getvalue())
        response.headers['Content-Type'] = 'application/zip'
        response.headers['Content-Disposition'] = f'attachment; filename="urbanflow_charts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/api/export/<export_type>', methods=['POST'])
def handle_export(export_type):
    """Handle consolidated export requests without legacy delegation."""
    try:
        payload = request.get_json() or {}
        format_type = (payload.get('format') or request.args.get('format') or '').lower()

        if export_type == 'predictions':
            rows = history_rows(filtered_history('all'))
            if format_type == 'json':
                response = make_response(predictions_to_json(rows))
                response.headers['Content-Type'] = 'application/json'
                response.headers['Content-Disposition'] = f'attachment; filename="predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
                return response

            response = make_response(predictions_to_csv(rows))
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename="predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
            return response

        if export_type == 'metrics':
            metrics = calculate_model_metrics()
            if format_type == 'pdf':
                return _metrics_pdf_response(metrics)
            return jsonify({"success": True, "metrics": metrics})

        if export_type == 'charts':
            if format_type == 'zip':
                return export_all_charts()
            return export_chart('feature_importance')

        if export_type in ('all', 'report'):
            rows = history_rows(filtered_history('all'))
            metrics = calculate_model_metrics()
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('predictions.csv', predictions_to_csv(rows))
                archive.writestr('metrics.json', json.dumps(metrics, indent=2, default=str))
                archive.writestr('README.txt', f'UrbanFlow complete export\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\n')

            response = make_response(zip_buffer.getvalue())
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Content-Disposition'] = f'attachment; filename="urbanflow_complete_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'
            return response

        return jsonify({"success": False, "error": "Unsupported export type"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
