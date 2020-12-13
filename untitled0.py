from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
print(spark)
print(spark.version)
print(spark.catalog.listTables())

dfb = spark.read.option('header','true').option('inferSchema','true').csv("E:\\Phan mem\\cuoc thi isds\\train_full.csv")

dfb.printSchema()
print(dfb.columns)
#dfb.Open.show(10)
dfb.select(['_c0', 'Open', 'High', 'Low', 'Close', 'Volume', 'body', 'upper_tail', 'lower_tail', 'SMA_50', 'SMA_20', 'ATR', 'CCI', 'SAR', 'hour', 'min', 'dayofweek', 'JPY', 'AUD', 'EUR', 'GBP', 'USD', 'lag_return_1', 'return_2', 'lag_return_2', 'return_3', 'lag_return_3', 'return_4', 'lag_return_4', 'return_5', 'lag_return_5', 'return_6', 'lag_return_6', 'return_7', 'lag_return_7', 'return_8', 'lag_return_8', 'return_9', 'lag_return_9', 'return_10', 'lag_return_10', 'return_11', 'lag_return_11', 'return_12', 'lag_return_12']).describe().show()




a=dfb.limit(5).toPandas()
a=dfb.toPandas()

dfb.select(['_c0', 'Open', 'High', 'Low', 'Close', 'Volume', 'body']).toPandas().hist()

abc = dfb.join(dfb,on='_c0')
abc.printSchema()


df = spark.read.option('header','true').option('inferSchema','true').csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.columns)

df.show()
df.printSchema()
df.select("BusinessTravel").show()


df.groupBy("BusinessTravel").count().show()

df.select(['Age','Attrition','PercentSalaryHike']).describe().show()

df.selectExpr("percentile_approx(Age,array(0,0.25,0.5,0.75,1)) as Age").show()


a = df.orderBy('Age',ascending = False).limit(10).toPandas()[['Age','Attrition','BusinessTravel','DistanceFromHome','EmployeeNumber']]


df.select(['Age']).toPandas().hist()


df1=df.drop('Over18')
df1.printSchema()

df[df.BusinessTravel == 'Travel_Rarely'].show(2)

import pyspark.sql.functions as F
df1=df.withColumn('cond', F.when(df.BusinessTravel == 'Travel_Frequently',1).when(df.BusinessTravel == 'Non-Travel',2).otherwise(3)).select('cond')

df1.groupBy("cond").count().show()

from pyspark.sql.types import DoubleType
fn = F.udf(lambda x: x+2, DoubleType())
df2 = df1.withColumn('cond2',fn(df1.cond))
df1.printSchema()
df1.show()
df2.printSchema()
df2.show()
df2.groupBy("cond2").count().show()

test = df1.groupBy("cond").count()

test.printSchema()

test2 = df1.join(test,on='cond')

test2.printSchema()
test2.show()

#import pyspark
#from pyspark import SparkContext
#sc =SparkContext()
#
## Below code is Spark 2+
#spark = pyspark.sql.SparkSession.builder.appName('test').getOrCreate()
#
#spark.range(10).collect()


from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# spark is an existing SparkSession
df = spark.read.json("C:\\Users\\Lenovo ThinkPad\\Desktop\\DS082020/people.json")
# Displays the content of the DataFrame to stdout
df.show()
# +----+-------+
# | age|   name|
# +----+-------+
# |null|Michael|
# |  30|   Andy|
# |  19| Justin|
# +----+-------+


# spark, df are from the previous example
# Print the schema in a tree format
df.printSchema()
# root
# |-- age: long (nullable = true)
# |-- name: string (nullable = true)

# Select only the "name" column
df.select("name").show()
# +-------+
# |   name|
# +-------+
# |Michael|
# |   Andy|
# | Justin|
# +-------+

# Select everybody, but increment the age by 1
df.select(df['name'], df['age'] + 1).show()
# +-------+---------+
# |   name|(age + 1)|
# +-------+---------+
# |Michael|     null|
# |   Andy|       31|
# | Justin|       20|
# +-------+---------+

# Select people older than 21
df.filter(df['age'] > 21).show()
# +---+----+
# |age|name|
# +---+----+
# | 30|Andy|
# +---+----+

# Count people by age
df.groupBy("age").count().show()
# +----+-----+
# | age|count|
# +----+-----+
# |  19|    1|
# |null|    1|
# |  30|    1|
# +----+-----+

df.select("age", "name").write.save("people.parquet", format="parquet")
df.write.parquet("people.parquet")

type(df)
df.show()
