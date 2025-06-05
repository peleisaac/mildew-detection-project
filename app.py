from app_pages.multipage import MultiPage
from app_pages.page_predict import page_mildew_detector
from app_pages.page_visual_study import page_visual_study_body
from app_pages.page_ml_performance import page_ml_performance_body
from app_pages.page_summary import page_summary
from app_pages.page_batch_analysis import page_batch_analysis

app = MultiPage(app_name="Mildew Detector")

app.add_page("ℹ️ Summary", page_summary)
app.add_page("🔍 Predict", page_mildew_detector)
app.add_page("🧬 Visual Study", page_visual_study_body)
app.add_page("📊 ML Performance", page_ml_performance_body)
app.add_page("🗺️ Batch Analysis", page_batch_analysis)

app.run()
