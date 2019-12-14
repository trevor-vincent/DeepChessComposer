#!/usr/bin/env python
import sys
import os
import time
import re
import numpy as np
import string
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import multiprocessing
import datetime
import time
import pyautogui

# from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType


#Size(width=3286, height=1050)
#( 61,  81, 18X: 1527 Y:  361 RGB: ( 61,  81, 18X: 1527 Y:  361 RGB: ( 61,  81, 18X: 1527 Y:  361 RGB: ( 61,  81, 18X: 1527 Y:  361 RGB: ( 61,  81, 18X: 1527 
#296 RGB: ( 61,  81, 18X: 1524 Y:  296 RGB: ( 61,  81, 18X: 1524 Y:  296 RGB: ( 61,  81, 18X: 1524 Y:  296 RGB: ( 61,  81, 18X: 1524 Y:  296 RGB: ( 61,  81, 18X: 1524 Y:  296 RGB: ( 61,  81, 18X: 1524 Y:  296 R

# prox = Proxy()
# prox.proxy_type = ProxyType.MANUAL
# prox.http_proxy = "138.68.53.44:8118"
# prox.socks_proxy = "138.68.53.44:8118"
# prox.ssl_proxy = "138.68.53.44:8118"

# capabilities = webdriver.DesiredCapabilities.CHROME
# prox.add_to_capabilities(capabilities)

# driver = webdriver.Chrome(desired_capabilities=capabilities)
# 138.68.53.44:8118

from selenium.webdriver.support.ui import WebDriverWait


def get_clear_browsing_button(driver):
    """Find the "CLEAR BROWSING BUTTON" on the Chrome settings page."""
    return driver.find_element_by_css_selector('* /deep/ #clearBrowsingDataConfirm')


def clear_cache(driver, timeout=60):
    """Clear the cookies and cache for the ChromeDriver instance."""
    # navigate to the settings page
    driver.get('chrome://settings/clearBrowserData')

    # wait for the button to appear
    wait = WebDriverWait(driver, timeout)
    wait.until(get_clear_browsing_button)

    # click the button to clear the cache
    get_clear_browsing_button(driver).click()

    # wait for the button to be gone before returning
    wait.until_not(get_clear_browsing_button)

def get_tags(puzzle_num):
    baseurl = "https://beta.chesstempo.com/chess-problems/" + str(puzzle_num)
    webdriver1 = webdriver.Chrome()
    # webdriver1.delete_all_cookies()
    # clear_cache(webdriver1)


    # prox = Proxy()
    # prox.proxy_type = ProxyType.MANUAL
    # prox.http_proxy = "138.68.53.44:8118"
    # prox.socks_proxy = "138.68.53.44:8118"
    # prox.ssl_proxy = "138.68.53.44:8118"

    # capabilities = webdriver.DesiredCapabilities.CHROME
    # prox.add_to_capabilities(capabilities)

    # webdriver1 = webdriver.Chrome(desired_capabilities=capabilities)



    webdriver1.get(baseurl)
    time.sleep(10)

    for i in range(0,20):
        num = 88 + i
        str1 = "//*[@id='label-text_ct-" + str(num) + "']"
        try:
            # print(str1)
            print(webdriver1.find_element_by_xpath("//*[@id='label-text_ct-" + str(num) + "']").text)
        except:
            pass
#1668 410
#464

#1532 377 
#    try:
 #   time.sleep(10)
#    pyautogui.click(x=18, y=1527)
#    pyautogui.moveTo(1527,428, duration=5)
#    pyautogui.click()
#    pyautogui.moveTo(1532,377, duration=5)
#    pyautogui.click()
#    while(1):
#        print("stopping")
 #webdriver1.find_element_by_xpath("/html/body/div[3]/main/chess-tactics/div[1]/div/div[3]/div[2]/board-controls/div[1]/fab-menu/nav/span[1]").click()
    time.sleep(4)
#    webdriver1.find_element_by_xpath("//*[@id='ct-116']").click()
    #webdriver1.find_element_by_xpath("//*[@id='ct-116']/i").click()
#    webdriver1.execute_script("document.querySelector(\"#ct-116\").click()")
 #   time.sleep(10)
  #  print(webdriver1.find_element_by_xpath("//*[@id='pgnField']").text)
#  except:
    #    pass
    # print(webdriver1.find_element_by_name("label"))
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='psramusernamepopup']").send_keys("majestichippo")
    # webdriver1.find_element_by_xpath("//*[@id='psramloginformpopup']/div[2]/div/div/section/div[1]/div[2]/div/input").send_keys("a1234!@#$")
    # webdriver1.find_element_by_xpath("//*[@id='psramloginformpopup']/div[2]/div/div/section/button").click()
    # time.sleep(10)
    # webdriver1.find_element_by_xpath("//*[@id='currencySwitchPM']").click()
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='h5c-hamburger-btn']/div").click()
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='drawerQuickSeat']").click()
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='qsCarouselItem6']/qs-back-common").click()
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='qsCarouselItem7']/qs-back-common").click()
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='qsCarouselItem0']/qs-back-common").click()
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='carouselItem0-playNow']").click()
    # time.sleep(2)
    # webdriver1.find_element_by_xpath("//*[@id='modalBtnTXTWBCLI_COMMON_OK']").click()
    # while(1):
        # time.sleep(10)

    webdriver1.quit()

        # webdriver1.save_screenshot(str(time.time()) + ".png")


get_tags(1)
