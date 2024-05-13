"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os

class Config():
    def __init__(self) -> None:
        
        # current working directory
        self.CWD = os.getcwd() # get current working directory