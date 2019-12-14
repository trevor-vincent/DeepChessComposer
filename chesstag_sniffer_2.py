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


def get_tags(puzzle_num):
    baseurl = "https://chesstempo.com/chess-problems/" + str(puzzle_num)
    webdriver1 = webdriver.Chrome()
    # webdriver1.delete_all_cookies()
    # clear_cache(webdriver1)


    webdriver1.get(baseurl)
    time.sleep(5)
    webdriver1.find_element_by_xpath("//*[@id='showFenButton']").click()
    time.sleep(5)
    print(webdriver1.find_element_by_xpath("//*[@id='pgnText']").text)

//*[@id="usernameField"]
//*[@id="passwordField"]
//*[@id="loginButton"]

get_tags(1)

