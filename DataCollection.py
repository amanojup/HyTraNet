# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:50:18 2021

@author: DELL
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from PIL import Image
import cv2
#from io import BytesIO
#options = webdriver.ChromeOptions()
#options.headless=True
import datetime

def current_time():
    return datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    #return datetime.datetime.now().strftime("%m_%S")

#PATH = "C:\Program Files (x86)\chromedriver.exe"
#DRIVER = 'chromedriver'
for timestep in range(126):
 driver=webdriver.Chrome(ChromeDriverManager().install())   
 #driver = webdriver.Chrome(PATH) #Load web driver
 # Load website
 driver.get('https://www.google.com/maps/@28.655785,77.2192958,15z/data=!5m1!1e1')
 print(driver.title)
 driver.fullscreen_window()
 sleep(5) #Wait for website to load
 #driver.execute_script("document.body.style_zoom='20%'")
 for zoom in range (2):
  driver.find_element(By.XPATH,'//*[@id="scene"]/div[3]/canvas').click()
 sleep(5)
 
 #driver.save_screenshot("D:\8_Article2021_ImageBased\Imagedata\Delhi_"+current_time()+".png")
 driver.get_screenshot_as_file('D:\8_Article2021_ImageBased\ImageBasedPaper\GoogleImage\my_screenshot.png')
 #driver.get_screenshot_as_file("D:\8_Article2021_ImageBased\Imagedata\Delhi_"+current_time()+".png")
 driver.quit()
 sleep(2)
 im = Image.open('D:\8_Article2021_ImageBased\ImageBasedPaper\GoogleImage\my_screenshot.png')
 left = 400
 top = 200
 right = 1600
 bottom = 800 
 im1 = im.crop((left, top, right, bottom))
 im1.save("D:\8_Article2021_ImageBased\ImageBasedPaper\GoogleImage\Delhi_"+current_time()+".png")
# im1.save('{}.png'.format(path, timestep))
 #im1.show()

