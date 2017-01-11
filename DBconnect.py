import MySQLdb

def connection():
    #conn = MySQLdb.connect(host="www.anis.tunisia-webhosting.com",
     #                       user="anis_ahmed",
      #                      passwd="zB2}hg!fRr56",
       #                     db="anis_android_api")
    conn = MySQLdb.connect(host="localhost",
                           user="root",
                           passwd="root",
                           db="therapio_api",
                           charset='utf8')
    c = conn.cursor()
    return c, conn