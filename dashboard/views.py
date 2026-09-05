import os
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.urls import reverse
from .forms import CohortUploadForm
from .models import Cohort, PredictionResult
from ml_pipeline.predict import run_prediction_pipeline


def upload_cohort(request):
    """Handle CSV upload, trigger preprocessing, model scoring and SHAP generation."""
    if request.method == "POST":
        form = CohortUploadForm(request.POST, request.FILES)
        if form.is_valid():
            cohort = form.save()
            # Run the ML pipeline (synchronous for simplicity)
            csv_path = cohort.csv_file.path
            # This function returns a DataFrame with predictions and SHAP dict per row
            results_df = run_prediction_pipeline(csv_path)
            # Persist results to DB
            for _, row in results_df.iterrows():
                PredictionResult.objects.create(
                    cohort=cohort,
                    record_id=row.get('record_id', ''),
                    churn_prob=row.get('churn_prob'),
                    lead_score=row.get('lead_score'),
                    risk_level=row.get('risk_level'),
                    shap_values=row.get('shap_values'),
                )
            return redirect(reverse('dashboard:results', args=[cohort.id]))
    else:
        form = CohortUploadForm()
    return render(request, "dashboard/upload.html", {"form": form})


def results_view(request, cohort_id):
    """Display a table of predictions with filters and SHAP popovers."""
    cohort = get_object_or_404(Cohort, pk=cohort_id)
    results = cohort.results.all()
    # Simple risk filter via query param
    risk_filter = request.GET.get('risk')
    if risk_filter:
        results = results.filter(risk_level=risk_filter)
    return render(request, "dashboard/results.html", {
        "cohort": cohort,
        "results": results,
        "risk_filter": risk_filter or "",
    })


def download_report(request, cohort_id):
    """Export predictions + SHAP values as CSV (Excel optional)."""
    cohort = get_object_or_404(Cohort, pk=cohort_id)
    qs = cohort.results.all()
    df = pd.DataFrame(list(qs.values(
        "record_id",
        "churn_prob",
        "lead_score",
        "risk_level",
        "shap_values",
    )))
    # Flatten SHAP JSON dict into separate columns (optional basic view)
    shap_expanded = df["shap_values"].apply(lambda x: x if isinstance(x, dict) else {})
    shap_df = pd.json_normalize(shap_expanded)
    df = pd.concat([df.drop(columns=["shap_values"]), shap_df], axis=1)
    # Write to temporary CSV in media folder
    out_path = os.path.join(settings.MEDIA_ROOT, f"report_cohort_{cohort.id}.csv")
    df.to_csv(out_path, index=False)
    from django.http import FileResponse
    return FileResponse(open(out_path, 'rb'), as_attachment=True, filename=f"cohort_{cohort.id}_report.csv")
