from app_pages.multipage import MultiPage
from app_pages import (
    page_predict,
    page_visual_study,
    page_ml_performance,
    page_business_impact,
    page_summary
)

app = MultiPage(app_name="🌿 SmartLeaf Dashboard")

app.add_page("🔍 Predict", page_predict)
app.add_page("🧬 Visual Study", page_visual_study)
app.add_page("📊 ML Performance", page_ml_performance)
app.add_page("💼 Business Impact", page_business_impact)
app.add_page("ℹ️ Summary", page_summary)

app.run()
