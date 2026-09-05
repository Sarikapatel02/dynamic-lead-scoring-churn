from django import forms
from .models import Cohort

class CohortUploadForm(forms.ModelForm):
    """Form for uploading a CSV cohort file."""
    class Meta:
        model = Cohort
        fields = ['name', 'csv_file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Cohort name'}),
            'csv_file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }
