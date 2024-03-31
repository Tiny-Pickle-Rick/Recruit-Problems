with open("./secret.daz","r",encoding="utf-8") as f:
    s=f.read()
message="".join([chr(int(x,16)) for x in s.split('X') if x])
cnt=0
for line in message.split():
    cnt=cnt+len(line)
with open("./interpretation.txt","w",encoding="utf-8") as result:
    result.write(f"{message}\n<解密人>2023090908023<情报总字数>{cnt}")