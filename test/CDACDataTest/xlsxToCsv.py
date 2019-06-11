import xlrd
import csv

def csv_from_excel():
    wb = xlrd.open_workbook('newData.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    your_csv_file = open('newData.csv', 'w', encoding = 'utf8')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()

# runs the csv_from_excel function:
csv_from_excel()