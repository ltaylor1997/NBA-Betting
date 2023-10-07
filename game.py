# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:33:12 2022

@author: lucas
"""

import os.path
from bs4 import BeautifulSoup
import requests
import re

html_front = "https://www.basketball-reference.com"

print("dfjkhgjdsg;klsdg")

def open_or_create_file():
    
    if os.path.exists("nba_game_logs.txt") is True:
      #  pass
        file = open("nba_game_logs.txt", "r")
        return file.read().split()
    else:
        
        f = open("nba_game_logs.txt", "w")
        for i in range(2000,2024,1):
            print(i)
            html = requests.get(f"https://www.basketball-reference.com/leagues/NBA_{i}_standings.html").text
            soup = BeautifulSoup(html, "html.parser")
            a_list = (soup.find_all("a"))
            href_list = [x.get('href') for x in a_list]
            
            teams = list(set([x for x in href_list if f"/{i}.html" in x]))
            print(teams)
            game_logs = [re.sub(f"{i}", f"{i}/gamelog", x) for x in teams]
            game_logs = [re.sub(".html", '', x) for x in game_logs]
            full_gl_link = [(html_front+x) for x in game_logs]
            full_gl_link.sort()
            for x in full_gl_link:
                f.write((x + " "))
        
        f = open("nba_game_logs.txt", "r")
        return f.read().split()
                

if __name__ == "__main__":
    
    f = open_or_create_file()
    print(f)
