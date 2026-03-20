# app/email_service.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
import json
from datetime import datetime

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.sender_email = os.getenv('SENDER_EMAIL', '')
        self.sender_password = os.getenv('SENDER_PASSWORD', '')
        
    def send_analytics_report(self, email_data, chart_images=None):
        """Send analytics report email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(email_data['recipients'])
            msg['Subject'] = email_data['subject']
            
            # Add email body
            body = email_data.get('message', 'Please find attached the analytics report.')
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments based on selections
            if email_data.get('includePDF', False):
                self.attach_pdf_report(msg, email_data)
                
            if email_data.get('includeCSV', False):
                self.attach_csv_data(msg, email_data)
                
            if email_data.get('includeCharts', False) and chart_images:
                self.attach_chart_images(msg, chart_images)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                
            return {'success': True, 'message': 'Email sent successfully'}
            
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def attach_pdf_report(self, msg, email_data):
        """Create and attach PDF report"""
        # This would generate a PDF report
        # For now, create a simple text file
        pdf_content = f"""
        UrbanFlow Analytics Report
        ===========================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        SUMMARY:
        - Model Accuracy: {email_data.get('accuracy', 'N/A')}%
        - Total Predictions: {email_data.get('total_predictions', 'N/A')}
        - Average Error: {email_data.get('avg_error', 'N/A')} bikes
        
        RECOMMENDATIONS:
        1. Continue monitoring model performance
        2. Consider retraining with new seasonal data
        3. Expand feature set for improved accuracy
        """
        
        attachment = MIMEApplication(pdf_content.encode('utf-8'), Name='analytics_report.txt')
        attachment['Content-Disposition'] = 'attachment; filename="analytics_report.txt"'
        msg.attach(attachment)
    
    def attach_csv_data(self, msg, email_data):
        """Attach CSV data"""
        # Sample CSV data - in real implementation, fetch from database
        csv_content = """timestamp,prediction_id,bikes_predicted,temperature,humidity
2024-01-01 08:00:00,1,1250,15.5,65
2024-01-01 09:00:00,2,1420,16.2,63
2024-01-01 10:00:00,3,980,17.8,60
"""
        
        attachment = MIMEApplication(csv_content.encode('utf-8'), Name='predictions_data.csv')
        attachment['Content-Disposition'] = 'attachment; filename="predictions_data.csv"'
        msg.attach(attachment)
    
    def attach_chart_images(self, msg, chart_images):
        """Attach chart images"""
        # Attach chart images to the email
        for chart_name, chart_data in chart_images.items():
            chart_info = f"Chart: {chart_name}\nGenerated: {datetime.now()}"
            attachment = MIMEApplication(chart_info.encode('utf-8'), Name=f'{chart_name}.txt')
            attachment['Content-Disposition'] = f'attachment; filename="{chart_name}.txt"'
            msg.attach(attachment)

# Singleton instance
email_service = EmailService()