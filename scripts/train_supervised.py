import os 
import sys 

DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIRECTORY + "/../") 
sys.path.append(DIRECTORY + "/../") 

from src.create_dataset import * 
