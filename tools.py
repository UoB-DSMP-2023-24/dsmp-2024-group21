import os
import re
import config

def getDates(path):
    dates=[]
    file_pattern = re.compile(r'^UoB_Set01_(\d{4}-\d{2}-\d{2}).*$')
    processed_files_path = os.path.join(path, 'processed.txt')
    with open(processed_files_path,'r') as f:
        processed_files = set(f.read().splitlines()) 
        for file in processed_files:
            match=file_pattern.match(file)
            date_str = match.group(1)
            dates.append(date_str)
            
    dates.sort()
    print(dates[:5])
    return dates
    
# getDates(config.LOBs_directory_path)
# getDates(config.Tapes_directory_path)
    