"""
Export Blueprint - Data export and email routes.
Routes: /export/*, /api/export/*
"""

from flask import Blueprint, request, jsonify, make_response
import io
import json
import os
import zipfile
import base64
from datetime import datetime

from app.blueprints.analytics import _loaded_models, get_model_meta
from app.email_service import email_service
from app.model_metrics import calculate_model_metrics, get_model_catalog
from app.models.loader import get_state
from app.shap_analysis import get_or_generate_shap_analysis
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


def _chart_payload(chart_type, time_filter='all', selected_model=None, shap_model=None):
    state = get_state()

    if chart_type == 'feature_importance':
        search_order = ['xgboost', 'stacking', 'lightgbm', 'advanced', 'improved', 'original']
        model = None
        source = 'original'
        if selected_model:
            requested = str(selected_model).strip().lower()
            candidate = state.get(f'{requested}_model')
            if candidate and hasattr(candidate, 'feature_importances_') and hasattr(candidate, 'feature_names_in_'):
                model = candidate
                source = requested
        for key in search_order:
            if model is not None:
                break
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
            if hour is None or value is None:
                continue
            try:
                hour_int = int(float(hour))
                demand = float(value)
            except (TypeError, ValueError):
                continue
            if 0 <= hour_int <= 23:
                hourly[hour_int].append(demand)
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

    if chart_type == 'shap_importance':
        selected_shap_model = str(shap_model or selected_model or 'lightgbm').strip().lower()
        shap_payload = get_or_generate_shap_analysis(selected_shap_model)
        shap_features = shap_payload.get('features', []) if shap_payload.get('success') else []
        if not shap_features and selected_shap_model != 'lightgbm':
            shap_payload = get_or_generate_shap_analysis('lightgbm')
            shap_features = shap_payload.get('features', []) if shap_payload.get('success') else []
        if not shap_features:
            return None

        top = shap_features[:12]
        return {
            'type': 'bar',
            'data': {
                'labels': [item.get('feature', 'Unknown') for item in top],
                'datasets': [{
                    'label': f"SHAP Importance ({shap_payload.get('source', 'LightGBM')})",
                    'data': [round(float(item.get('importance', 0.0)), 2) for item in top],
                    'backgroundColor': '#2563eb',
                    'borderColor': '#1d4ed8',
                }],
            },
            'options': {'title': 'SHAP Feature Importance', 'ylabel': 'Mean |SHAP|'},
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
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    styles = getSampleStyleSheet()

    story = [
        Paragraph("UrbanFlow Model Metrics Report", styles['Title']),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
        Spacer(1, 20),
    ]

    model_catalog = get_model_catalog()
    story.append(Paragraph("All Model Performance", styles['Heading2']))
    story.append(Spacer(1, 8))

    model_order = ['original', 'improved', 'advanced', 'xgboost', 'lightgbm', 'stacking', 'unified']
    report_rows = [['Model', 'R2', 'RMSE', 'MAE', 'Features']]

    for model_key in model_order:
        model = model_catalog.get(model_key)
        if not model:
            continue
        report_rows.append([
            model.get('name', model_key),
            f"{float(model.get('r2', 0.0)):.3f}",
            f"{float(model.get('rmse', 0.0)):.1f}",
            f"{float(model.get('mae', 0.0)):.1f}",
            str(model.get('features', 0)),
        ])

    if len(report_rows) > 1:
        all_models_table = Table(
            report_rows,
            colWidths=[4.0 * inch, 1.1 * inch, 1.4 * inch, 1.4 * inch, 1.1 * inch],
            repeatRows=1,
        )
        all_models_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.75, colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ]))
        story.append(all_models_table)

    doc.build(story)

    response = make_response(buffer.getvalue())
    response.headers["Content-Disposition"] = f'attachment; filename="urbanflow_metrics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
    response.headers["Content-Type"] = "application/pdf"
    return response


@export.route('/export/predictions/csv', methods=['GET'])
def export_predictions_csv():
    """Export prediction history as CSV."""
    try:
        time_filter = request.args.get('filter', 'all')
        rows = history_rows(filtered_history(time_filter))
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
        time_filter = request.args.get('filter', 'all')
        rows = history_rows(filtered_history(time_filter))
        response = make_response(predictions_to_json(rows))
        response.headers["Content-Disposition"] = "attachment; filename=predictions_history.json"
        response.headers["Content-Type"] = "application/json"
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/export/model-report', methods=['GET'])
def export_model_report():
    """Export a model performance report as HTML."""
    try:
        catalog = get_model_catalog()
        model_order = ['original', 'improved', 'advanced', 'xgboost', 'lightgbm', 'stacking', 'unified']
        rows = [catalog[key] for key in model_order if key in catalog]
        rows.sort(key=lambda row: float(row.get('r2', 0.0)), reverse=True)

        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        table_rows = []
        for idx, model in enumerate(rows, start=1):
            table_rows.append(
                f"""
                <tr>
                    <td>{idx}</td>
                    <td>{model.get('name', 'Unknown')}</td>
                    <td>{float(model.get('r2', 0.0)):.3f}</td>
                    <td>{float(model.get('rmse', 0.0)):.1f}</td>
                    <td>{float(model.get('mae', 0.0)):.1f}</td>
                    <td>{int(model.get('features', 0))}</td>
                    <td>{model.get('recommended_for', 'General use')}</td>
                </tr>
                """
            )

        best_model_name = rows[0].get('name', 'Unknown') if rows else 'Unknown'
        best_model_r2 = f"{float(rows[0].get('r2', 0.0)):.3f}" if rows else "0.000"

        report_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>UrbanFlow Model Report</title>
  <style>
    body {{
      font-family: "Segoe UI", Tahoma, sans-serif;
      margin: 24px;
      color: #0f172a;
      background: #f8fafc;
    }}
    .container {{
      max-width: 1080px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      color: #2563eb;
    }}
    .meta {{
      color: #475569;
      margin-bottom: 18px;
      font-size: 14px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 20px;
    }}
    .card {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 12px;
    }}
    .card .label {{
      color: #334155;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card .value {{
      margin-top: 4px;
      font-size: 18px;
      font-weight: 700;
      color: #0f172a;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #e2e8f0;
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #2563eb;
      color: white;
    }}
    tr:nth-child(even) td {{
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>UrbanFlow Model Performance Report</h1>
    <div class="meta">Generated: {generated_at}</div>
    <div class="summary">
      <div class="card"><div class="label">Models Covered</div><div class="value">{len(rows)}</div></div>
      <div class="card"><div class="label">Best Model</div><div class="value">{best_model_name}</div></div>
      <div class="card"><div class="label">Best R2</div><div class="value">{best_model_r2}</div></div>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Model</th>
          <th>R2</th>
          <th>RMSE</th>
          <th>MAE</th>
          <th>Features</th>
          <th>Recommended Use</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
        """

        response = make_response(report_html)
        response.headers["Content-Disposition"] = "attachment; filename=model_report.html"
        response.headers["Content-Type"] = "text/html; charset=utf-8"
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
        selected_model = request.args.get('model')
        selected_shap_model = request.args.get('shap_model')
        if format_type not in ('png', 'pdf', 'svg', 'jpg'):
            return jsonify({"success": False, "error": "Invalid format"}), 400

        chart_data = _chart_payload(chart_type, time_filter, selected_model=selected_model, shap_model=selected_shap_model)
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
        selected_model = request.args.get('model')
        selected_shap_model = request.args.get('shap_model')
        if format_type not in ('png', 'pdf'):
            return jsonify({"success": False, "error": "Invalid format"}), 400

        zip_buffer = io.BytesIO()
        exported = []
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for chart_type in ('feature_importance', 'shap_importance', 'performance_radar', 'hourly_patterns'):
                chart_data = _chart_payload(
                    chart_type,
                    time_filter,
                    selected_model=selected_model,
                    shap_model=selected_shap_model,
                )
                if not chart_data:
                    continue
                archive.writestr(
                    f'urbanflow_{chart_type}.{format_type}',
                    _generate_chart_image(chart_data, format_type),
                )
                exported.append(chart_type)
            archive.writestr(
                'README.txt',
                (
                    f'UrbanFlow chart export\n'
                    f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                    f'Filter: {time_filter}\n'
                    f'Charts exported: {", ".join(exported) if exported else "none"}\n'
                ),
            )

        response = make_response(zip_buffer.getvalue())
        response.headers['Content-Type'] = 'application/zip'
        response.headers['Content-Disposition'] = f'attachment; filename="urbanflow_charts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/api/export/all-charts-exact', methods=['POST'])
def export_all_charts_exact():
    """Export exactly what is currently rendered on the dashboard charts."""
    try:
        payload = request.get_json() or {}
        chart_images = payload.get('charts', [])
        if not isinstance(chart_images, list) or not chart_images:
            return jsonify({"success": False, "error": "No chart images provided"}), 400

        zip_buffer = io.BytesIO()
        exported = []
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for item in chart_images:
                filename = str(item.get('filename', '')).strip()
                data_url = item.get('data_url')
                if not filename or not isinstance(data_url, str) or not data_url.startswith('data:image/'):
                    continue

                # Expected form: data:image/png;base64,XXXXX
                try:
                    _, encoded = data_url.split(',', 1)
                    binary = base64.b64decode(encoded)
                except Exception:
                    continue

                if not filename.lower().endswith('.png'):
                    filename = f"{filename}.png"
                archive.writestr(filename, binary)
                exported.append(filename)

            if not exported:
                return jsonify({"success": False, "error": "No valid chart images to export"}), 400

            archive.writestr(
                'README.txt',
                (
                    f'UrbanFlow exact chart export\n'
                    f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                    f'Charts exported: {", ".join(exported)}\n'
                    f'Source: client-rendered dashboard state\n'
                ),
            )

        response = make_response(zip_buffer.getvalue())
        response.headers['Content-Type'] = 'application/zip'
        response.headers['Content-Disposition'] = f'attachment; filename="urbanflow_charts_exact_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'
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
            filters = payload.get('filters') if isinstance(payload.get('filters'), dict) else {}
            time_filter = filters.get('time_filter') or request.args.get('filter', 'all')
            rows = history_rows(filtered_history(time_filter))
            metrics = calculate_model_metrics()
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('predictions.csv', predictions_to_csv(rows))
                archive.writestr('metrics.json', json.dumps(metrics, indent=2, default=str))
                archive.writestr(
                    'README.txt',
                    (
                        f'UrbanFlow complete export\n'
                        f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                        f'Filter: {time_filter}\n'
                        f'Predictions exported: {len(rows)}\n'
                    ),
                )

            response = make_response(zip_buffer.getvalue())
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Content-Disposition'] = f'attachment; filename="urbanflow_complete_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'
            return response

        return jsonify({"success": False, "error": "Unsupported export type"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
