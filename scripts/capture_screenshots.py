import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    page.goto("http://localhost:8501", timeout=30000)
    page.wait_for_timeout(25000)  # first load includes parquet + model load

    page.screenshot(path="/tmp/new_ui_01_overview.png", full_page=True)
    print("Captured overview")

    sidebar = page.get_by_test_id("stRadioGroup")
    sidebar.get_by_text("Data Exploration (EDA)", exact=True).click()
    page.wait_for_timeout(6000)
    page.mouse.wheel(0, 3000)
    page.wait_for_timeout(2000)
    page.mouse.wheel(0, -3000)
    page.wait_for_timeout(2000)
    page.screenshot(path="/tmp/new_ui_02_eda.png", full_page=True)
    print("Captured EDA")

    sidebar.get_by_text("Risk Prediction", exact=True).click()
    page.wait_for_timeout(2000)
    submit_btn = page.get_by_role("button", name="Score Applicant")
    submit_btn.scroll_into_view_if_needed()
    submit_btn.click()
    page.wait_for_timeout(4000)
    page.screenshot(path="/tmp/new_ui_03_prediction.png", full_page=True)
    print("Captured prediction with gauge")

    sidebar.get_by_text("Explainability", exact=True).click()
    page.wait_for_timeout(4000)
    page.screenshot(path="/tmp/new_ui_04_explainability.png", full_page=True)
    print("Captured explainability")

    sidebar.get_by_text("Business Rules", exact=True).click()
    page.wait_for_timeout(3000)
    page.screenshot(path="/tmp/new_ui_05_rules.png", full_page=True)
    print("Captured business rules")

    browser.close()


