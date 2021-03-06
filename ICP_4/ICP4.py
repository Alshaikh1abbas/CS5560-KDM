# import needed library for ICP4
import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkFiles

def main():
    sc = pyspark.SparkContext()

    # calling the text file from my laptop which has the context to do the word count
    file = sc.textFile('C:/Users/AB/Desktop/ICP_4/NewsFile.txt')
    counts = file.flatMap(lambda line: line.split(' ')).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

    # this function (x) to print the count results
    for x in counts.collect():
        print(x)

    sqlContext = SQLContext(sc)

    # calling the DATA.CSV file from my laptop
    file = sqlContext.read.csv(SparkFiles.get('C:/Users/AB/Desktop/ICP_4/data.csv'), header=True, inferSchema=True)

    # 1st Transformation : Printing 'senior citizen' only.
    ssCust = file.filter(file['SeniorCitizen'] == '1')

    # 2nd Transformation : Printing 'female customers' only.
    fCust = file.filter(file['gender'] == 'Female')

    # 3rd Transformation : Printing combination of 'senior citizen' & 'female customers'.
    fssCust = fCust.intersect(ssCust)

    # 1st Action: count the number of female customers only.
    print('There are {} female customers'.format(fCust.count()))

    # 2nd Action: count the number of female senior citizen customers
    print('There are {} female senior citizen customers'.format(fssCust.count()))

    # 3rd Action: show the average monthly charge for each gender of senior citizens
    ssCust.groupBy('gender').agg({'MonthlyCharges': 'avg', 'Gender': 'count'}).show()

    # Command to print the RDD schema
    file.printSchema()

    # Command to print the first 5 entries in an RDD
    fssCust.show(5)

    # Command to print customers by contract type
    file.groupBy('Contract').count().show()

    # Command to show count, avg, stdev, min & max of each column
    file.describe().show()