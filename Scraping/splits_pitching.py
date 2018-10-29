import pandas as pd
import requests
from time import sleep
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#from selenium.webdriver.common.keys import Keys
import re
#from selenium.webdriver.chrome.options import Options

options = webdriver.ChromeOptions()
options.add_extension('AdBlock_v3.27.0.crx')
driver = webdriver.Chrome('/Users/kweithers/Downloads/chromedriver',chrome_options= options)
url = 'https://www.baseball-reference.com/leagues/MLB/2014-standard-pitching.shtml'
driver.get(url)

master = []


def parse_batting_handedness(x):
     if 'Right' in str(x).split()[2]:
         return 'R'
     elif 'Left' in str(x).split()[2]:
         return 'L'
     else:
         return 'S'
def parse_throwing_handedness(x):
     if 'Right' in str(x).split()[5]:
         return 'R'
     elif 'Left' in str(x).split()[5]:
         return 'L'
     else:
         return 'S'
    
WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable(
                (By.XPATH, """//*[@id="players_standard_pitching"]/thead/tr/th[6]""")))

d = True
while d:
    try:
        driver.find_element_by_xpath("""//*[@id="players_standard_pitching"]/thead/tr/th[6]""").click()
        d = False
    except:
        sleep(5)
        print("Was a nice sleep, now let me continue...")
        continue

#loop through players
#num_rows = driver.find_elements_by_xpath("""//*[@id="players_standard_batting"]/tbody/tr/td[1]/a""")
for i in range(1,400):
    try:        
        name_field = driver.find_element_by_xpath("""//*[@id="players_standard_pitching"]/tbody/tr["""+str(i)+"""]/td[1]/a[1]""")
        name = name_field.text
    
        player_element = driver.find_element_by_xpath("""//*[@id="players_standard_pitching"]/tbody/tr["""+str(i)+"""]/td[1]""")
        player_id = player_element.get_attribute("data-append-csv")
    
    
        a = 'https://www.baseball-reference.com/players/split.fcgi?id=' + player_id + '&year=Career&t=p'
        
        #open tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(a)
        
        r = requests.get(driver.current_url)
    
        soup = BeautifulSoup(r.text,'lxml')    
        
        stats_html1 = soup.find(string=re.compile('id="total"'))
        if stats_html1 is not None:
            stats_soup1 = BeautifulSoup(stats_html1, "lxml")
            df1 = pd.read_html(str(stats_soup1))
        else:
            df1 = [pd.DataFrame()]
        
        stats_html2 = soup.find(string=re.compile('id="plato"'))
        if stats_html2 is not None:
            stats_soup2 = BeautifulSoup(stats_html2, "lxml")
            df2 = pd.read_html(str(stats_soup2))
        else:
            df2 = [pd.DataFrame()]
    
        stats_html3 = soup.find(string=re.compile('id="half"'))
        if stats_html3 is not None:
            stats_soup3 = BeautifulSoup(stats_html3, "lxml")
            df3 = pd.read_html(str(stats_soup3))
        else:
            df3 = [pd.DataFrame()]
    
        stats_html4 = soup.find(string=re.compile('id="hmvis"'))
        if stats_html4 is not None:
            stats_soup4 = BeautifulSoup(stats_html4, "lxml")
            df4 = pd.read_html(str(stats_soup4))
        else:
            df4 = [pd.DataFrame()]
    
        stats_html5 = soup.find(string=re.compile('id="traj"'))
        if stats_html5 is not None:
            stats_soup5 = BeautifulSoup(stats_html5, "lxml")
            df5 = pd.read_html(str(stats_soup5))
        else:
            df5 = [pd.DataFrame()]
    
    #    stats_html6 = soup.find(string=re.compile('id="gbfb"'))
    #    if stats_html6 is not None:
    #        stats_soup6 = BeautifulSoup(stats_html6, "lxml")
    #        df6 = pd.read_html(str(stats_soup6))
    #    else:
    #        df6 = [pd.DataFrame()]
    
        
        df = pd.concat([df1[0],df2[0],df3[0],df4[0],df5[0]])
    
    
        handedness_string = soup.find_all("p")[1]
        handedness = parse_throwing_handedness(handedness_string)
        
        master.append([name,handedness,df])
        #close tab
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
    except:
        continue
         
    
for c in range(len(master)):
    master[c][2]['name'] = master[c][0] 
    master[c][2]['hand'] = master[c][1]

y = []
for j in range(len(master)):
    y.append(master[j][2])

result = pd.concat(y)
result.to_csv('Pitching_splits_2014.csv',encoding='utf-8',index=False)