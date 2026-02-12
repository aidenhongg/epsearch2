
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import re

DOWNLOAD_PATH = "./database"
LINK = "https://journaliststudio.google.com/pinpoint/search?collection=c109fa8e7dcf42c1&p="

chrome_options = Options()
prefs = {"download.default_directory": DOWNLOAD_PATH}
chrome_options.add_experimental_option("prefs", prefs)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

menu_selector = 'div[aria-label="Top bar menu"]'

def download_pinpoint_files(driver, wait, count):
    filenames = []
    for i in range(count):
        current_elements = wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, 'div[data-tooltip*="pdf"]') if len(d.find_elements(By.CSS_SELECTOR, 'div[data-tooltip*="pdf"]')) >= count else False)
        target = current_elements[i]
        wait.until(EC.element_to_be_clickable(target))
        filenames.append(target.get_attribute("data-tooltip"))

        # 2. Click the element
        target.click()
        try:
            # This waits for a specific element that signals the file is "open"
            menu_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, menu_selector)))
            menu_button.click()

            download_xpath = "//span[contains(text(), 'Download transcript') or contains(text(), 'Download original file')]"
            download_btn = wait.until(EC.element_to_be_clickable((By.XPATH, download_xpath)))

            download_btn.click()
            driver.back()            

        except Exception as e:
            print(f"Error waiting for file to open: {e}")
            driver.back()
            continue
        
    return filenames

p = 1
with open("filenames.txt", "a") as f:
    while True:
        driver.get(LINK + str(p))
        wait = WebDriverWait(driver, 10)
        pagination_element = wait.until(lambda d: d.find_element(By.XPATH, "//*[contains(translate(text(), '0123456789,', ''), ' -  of ')]"))
        
        text = pagination_element.text 
        match = re.search(r'([\d,]+)\s*-\s*([\d,]+)', text)

        start_num = int(match.group(1).replace(',', ''))
        end_num = int(match.group(2).replace(',', ''))
        count = end_num - start_num + 1
        print(count)

        try:
            f.write("\n".join(download_pinpoint_files(driver, wait, count)) + "\n")

        except TimeoutException:
            print("No more files to download or page did not load properly.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
        p += 1