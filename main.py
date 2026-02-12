
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=1, help="Starting page number")
parser.add_argument("--limit", type=int, default=100, help="Ending page number")
args = parser.parse_args()

START = args.start
LIMIT = args.limit

DOWNLOAD_PATH = os.path.abspath("./database")
if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)
LINK = "https://journaliststudio.google.com/pinpoint/search?collection=c109fa8e7dcf42c1&p="

chrome_options = Options()
prefs = {"download.default_directory": DOWNLOAD_PATH}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

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
            with open("./badfiles.txt", "a") as fe:
                fe.write(f"{target.get_attribute('data-tooltip')}\n")

            driver.back()
            continue
        
    return filenames

with open("filenames.txt", "a") as f:
    while START <= LIMIT:
        driver.get(LINK + str(START))
        wait = WebDriverWait(driver, 10)
        pagination_element = wait.until(lambda d: d.find_element(By.XPATH, "//*[contains(translate(text(), '0123456789,', ''), ' -  of ')]"))
        
        text = pagination_element.text 
        match = re.search(r'([\d,]+)\s*-\s*([\d,]+)', text)

        start_num = int(match.group(1).replace(',', ''))
        end_num = int(match.group(2).replace(',', ''))
        count = end_num - start_num + 1

        try:
            f.write("\n".join(download_pinpoint_files(driver, wait, count)) + "\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            with open("./badpages.txt", "a") as fe:
                fe.write(f"Error on page {START}: {e}\n")
            
            continue
        START += 1