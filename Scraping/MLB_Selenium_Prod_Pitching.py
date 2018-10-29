import pandas as pd
import requests
from time import sleep
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#from selenium.webdriver.chrome.options import Options

options = webdriver.ChromeOptions()
options.add_extension('AdBlock_v3.27.0.crx')
#options.add_argument("headless")  
driver = webdriver.Chrome('/Users/kweithers/Downloads/chromedriver',chrome_options= options)
url = 'https://www.baseball-reference.com/leagues/MLB/2013-standard-pitching.shtml'
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
#num_rows = driver.find_elements_by_xpath("""//*[@id="players_standard_pitching"]/tbody/tr/td[1]/a""")
for i in range(1,400):        
#for i in range(len(num_rows)):    
    try:
        name_field = driver.find_element_by_xpath("""//*[@id="players_standard_pitching"]/tbody/tr["""+str(i)+"""]/td[1]/a[1]""")
        name = name_field.text
    
        player_element = driver.find_element_by_xpath("""//*[@id="players_standard_pitching"]/tbody/tr["""+str(i)+"""]/td[1]""")
        player_id = player_element.get_attribute("data-append-csv")
    
        a = 'https://www.baseball-reference.com/players/gl.fcgi?id=' + player_id + '&t=p&year=2013'
        
        #open tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(a)
        
        r = requests.get(driver.current_url)
    
        soup = BeautifulSoup(r.text,'lxml')          
    #   name = soup.find("h1", {"itemprop": "name"}).text            
        table = soup.find("table", {"id": "pitching_gamelogs"})
        
        handedness_string = soup.find_all("p")[1]
        handedness = parse_throwing_handedness(handedness_string)
        
        df = pd.read_html(str(table))
        master.append([name,handedness,df[0]])
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
result.to_csv('Pitching_2013.csv',encoding='utf-8',index=False)
