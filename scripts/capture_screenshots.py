import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto("http://localhost:8501", timeout=30000)
    page.wait_for_timeout(15000)  # first load includes 41MB parquet + model load

    # 1. Overview
    page.screenshot(path="documents/screenshots/01_overview.png", full_page=True)
    print("Captured 01_overview.png")

    # 2. EDA
    sidebar = page.get_by_test_id("stRadioGroup")
    sidebar.get_by_text("Data Exploration (EDA)", exact=True).click()
    page.wait_for_timeout(2500)
    page.screenshot(path="documents/screenshots/02_eda.png", full_page=True)
    print("Captured 02_eda.png")

    # 3. Risk Prediction - fill form and submit for a realistic screenshot
    sidebar.get_by_text("Risk Prediction", exact=True).click()
    page.wait_for_timeout(2000)
    page.screenshot(path="documents/screenshots/03_risk_prediction_form.png", full_page=True)
    print("Captured 03_risk_prediction_form.png")

    # Submit the form with defaults to get a scored result
    submit_btn = page.get_by_role("button", name="Score Applicant")
    submit_btn.scroll_into_view_if_needed()
    submit_btn.click()
    page.wait_for_timeout(5000)
    page.screenshot(path="documents/screenshots/03b_risk_prediction_result.png", full_page=True)
    print("Captured 03b_risk_prediction_result.png")

    # 4. Explainability (now that we've scored an applicant, this should show real content)
    sidebar.get_by_text("Explainability", exact=True).click()
    page.wait_for_timeout(4000)
    page.screenshot(path="documents/screenshots/04_explainability.png", full_page=True)
    print("Captured 04_explainability.png")

    # 5. Business Rules
    sidebar.get_by_text("Business Rules", exact=True).click()
    page.wait_for_timeout(2500)
    page.screenshot(path="documents/screenshots/05_business_rules.png", full_page=True)
    print("Captured 05_business_rules.png")

    # 6. Talk to Data
    sidebar.get_by_text("Talk to Data", exact=True).click()
    page.wait_for_timeout(2000)
    page.screenshot(path="documents/screenshots/06_talk_to_data.png", full_page=True)
    print("Captured 06_talk_to_data.png")

    browser.close()

