from django.db import models

class Cohort(models.Model):
    """Uploaded CSV representing a group of customers or leads."""
    name = models.CharField(max_length=100)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    csv_file = models.FileField(upload_to='cohorts/')

    def __str__(self):
        return self.name

class PredictionResult(models.Model):
    """Stores prediction and SHAP output for a single record."""
    cohort = models.ForeignKey(Cohort, on_delete=models.CASCADE, related_name='results')
    record_id = models.CharField(max_length=100)  # identifier from source CSV
    churn_prob = models.FloatField(null=True, blank=True)
    lead_score = models.FloatField(null=True, blank=True)
    risk_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Medium', 'Medium'), ('Low', 'Low')])
    shap_values = models.JSONField(null=True, blank=True)  # feature: contribution mapping
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.record_id} - {self.risk_level}"
