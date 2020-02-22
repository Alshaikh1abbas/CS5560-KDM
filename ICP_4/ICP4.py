pythonimport p

if not 'sc' in globals():
    sc = pyspark.SparkContext()
text_file = sc.textFile("B:/InTo/ICP_4/NewsFile.txt")
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
for x in counts.collect():
    print (x)
