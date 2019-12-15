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
from selenium.webdriver.chrome.options import Options



display = Display(visible=0, size=(800, 600))
display.start()


def get_tags(puzzle_num):

    wait_time = 1
    chrome_options = Options()
    #chrome_options.add_argument("--disable-extensions")
    #chrome_options.add_argument("--disable-gpu")

    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    
    chrome_options.add_argument("--headless")
    webdriver1 = webdriver.Chrome(options=chrome_options)

    baseurl = "https://chesstempo.com/chess-problems/" + str(puzzle_num)
    file_details = open(str(puzzle_num) + "_details.txt", 'w')
    file_moves = open(str(puzzle_num) + "_moves.txt", 'w')
    file_tags = open(str(puzzle_num) + "_tags.txt", 'w')
    file_pgn = open(str(puzzle_num) + ".pgn", 'w')

    webdriver1.get(baseurl)
    time.sleep(wait_time)
    webdriver1.find_element_by_xpath("//*[@id='usernameField']").send_keys("chessraptor")
    # time.sleep(wait_time)
    # time.sleep(wait_time)
    webdriver1.find_element_by_xpath("//*[@id='loginButton']").click()
    time.sleep(wait_time)
    webdriver1.get(baseurl)

    time.sleep(wait_time)
    file_details.write(webdriver1.find_element_by_xpath("//*[@id='problemDetailsDisplay']").text)
    file_moves.write(webdriver1.find_element_by_xpath("//*[@id='board-moves']").text)
    # time.sleep(wait_time)
    webdriver1.find_element_by_xpath("//*[@id='showFenButton']").click()
    time.sleep(wait_time)
    file_pgn.write(webdriver1.find_element_by_xpath("//*[@id='pgnText']").text)
    # time.sleep(wait_time)
    webdriver1.find_element_by_xpath("//*[@id='categoryVoteAgainstButton']").click()
    time.sleep(wait_time)
    file_tags.write(webdriver1.find_element_by_xpath("//*[@id='tagVote']").text)

    file_details.close()
    file_moves.close()
    file_tags.close()
    file_pgn.close()
    webdriver1.close()

#78000
    
# //*[@id="usernameField"]
# //*[@id="passwordField"]
# //*[@id="loginButton"]


for i in range(12, 78000):
    try:
        get_tags(i)
        print("Done " + str(i))
    except:
        pass



display.stop()
