import requests
import csv
import json

symbol = "MSFT"
url = "https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol="+symbol+"&apikey=3FEF7XKMWV6LPTFJ"
r = requests.get(url)
data = r.json()

quarterly_eps = data.get('quarterlyReports')

debtToEquityRatios = []

for quarter in quarterly_eps:
    longTermDebt = quarter.get('longTermDebt')
    if(longTermDebt == 'None'):
        longTermDebt = 0
    
    currentLongTermDebt = quarter.get('currentLongTermDebt')
    if(currentLongTermDebt == 'None'):
        currentLongTermDebt = 0
    
    
    total = int(longTermDebt) + int(currentLongTermDebt)
    ratio = total / int(quarter.get('totalShareholderEquity'))
    debtToEquityRatios.append({quarter.get('fiscalDateEnding'):ratio})
    
json_data = json.dumps(debtToEquityRatios)


with open("DTE_"+symbol+".json",'w') as f:
    json.dump(json_data,f)


#print(data)