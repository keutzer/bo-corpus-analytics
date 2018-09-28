url = 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'
import re
from bs4 import BeautifulSoup
import requests

resp = requests.get(url)
assert resp.status_code == 200

page = BeautifulSoup(resp.content)

html_tables = page.find_all("table")

dfs = []

for html_table in html_tables:
    html_rows = html_table.find_all("tr")
    headers = []
    rows = []
    for html_row in html_rows:
        header_cols = html_row.find_all("th")
        regular_cols = html_row.find_all("td")
        if not regular_cols:
            headers.append([x.text for x in header_cols])
        else:
            all_cols = html_row.find_all(["th", "td"])
            rows.append([x.text for x in all_cols])

    nhead = len(headers)
    nrows = len(rows)
    if nhead != 2 or nrows == 0:
        print('weird num headers {} num rows {}, skipping'.format(
            nhead, nrows))
        continue
    real_header = headers[0]
    print('*' * 60)
    print('table')
    print(real_header)
    print(rows)
