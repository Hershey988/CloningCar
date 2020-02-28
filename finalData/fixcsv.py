import csv

with open('driving_log.csv', 'r') as harsh:
	with  open('driving_log_new.csv', 'w',newline ='') as output:
		writer = csv.writer(output)
		read = csv.reader(harsh)
		count =0
		for row in read :
			count+=1
			if(count > 3) and (count<9576):
				writer.writerow(row)
harsh.close()
output.close()
