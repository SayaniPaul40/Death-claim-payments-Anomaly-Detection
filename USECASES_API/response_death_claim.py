import requests
import json
url = "http://10.240.21.72:5000/death_claim"
data = json.loads(input().strip())
#[{"due_yyyy": 2023, "due_mm": 6,"fup_mas_yyyy": 2001,"fup_mas_mm": 10,"cm_mode": 12,"claim_status": "P", "date_of_death": "4/25/2023","accounted_on": "09/11/2000 00:00","policy_term": 5,"ord_prem_flag": 0,"life_death_ind": 0,"docket_availability": "Y","premium": 10000,"paid_date": "6/9/2023 00:00","commencement_date": "09/10/1996","total_payment": 54441,"total_deduction": 1888,"claim_amount": 54317,"vested_bonus": 0,"interim_bonus": 0}]

response = requests.post(url,json=data)
print(response.json())
