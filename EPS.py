import requests
import csv
import json

symbol = "ADBE"
url = "https://www.alphavantage.co/query?function=EARNINGS&symbol="+symbol+"&apikey=3FEF7XKMWV6LPTFJ"
r = requests.get(url)
data = r.json()

quarterly_eps = data.get('quarterlyEarnings')


with open("EPS_"+symbol+".json",'w') as f:
    json.dump(data,f)


#print(data)