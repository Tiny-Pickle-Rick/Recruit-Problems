import datetime as dt
def date_now():
    today=input("请输入今天的日期(example:2022-09-09):")
    num=input("请问你要查询几个人:")
    num=int(num)
    return today,num
def date_input(person):
    k=input("请输入已经接种了几针:")
    k=int(k)
    v=input("请输入最近一次的接种日期(example:2022-09-09):")
    person.append({k:v})#key是接种针数，value是接种日期
    return
def date_output(today,person):
    L=[]
    td=dt.datetime.strptime(today, "%Y-%m-%d").date()
    for each in person:#each得到一个字典
        times=list(each.keys())[0]#针数
        last=list(each.values())[0]#上次接种日期
        lst = dt.datetime.strptime(last, "%Y-%m-%d").date()
        match times:
            case 0:
                L.append({True: today})
            case 1:
                if td-lst >= dt.timedelta(days=30):
                    L.append({True: today})
                else:
                    L.append({False: (lst+dt.timedelta(days=30)).strftime('%Y-%m-%d')})
            case 2:
                if td-lst >= dt.timedelta(days=180):
                    L.append({True: today})
                else:
                    L.append({False: (lst+dt.timedelta(days=180)).strftime('%Y-%m-%d')})
            case 3:
                L.append({False: " "})
    return L

today,num = date_now()
person=[]
while num>0:
    date_input(person)
    num=num-1
print(date_output(today,person))