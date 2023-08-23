



# docname =
# batchname = '[Batch 2399/2400]'
# picname = 'hrnet.svg'
# with open('nohup.out', 'r', encoding='utf-8') as f:
#     data = f.read().split('\n')
#     tmp = [i for i in data if '[Batch 1199/1200]' in i]
#     # Dlosslist = [float(m.split('] [')[2][-8:].replace(']', '')) for m in tmp]
#     Glosslist1 = [float(m.split('] [')[3][-8:].replace(']', '')) for m in tmp]

#     # print(Dlosslist)
#     # print(Glosslist)
with open('srgan_hrnet.log', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    tmp = [i for i in data if '[Batch 4799/4800]' in i]
    # Dlosslist = [float(m.split('] [')[2][-8:].replace(']', '')) for m in tmp]
    Glosslist2 = [float(m.split('] [')[3][-8:].replace(']', '')) for m in tmp]

with open('srgan.log', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    tmp = [i for i in data if '[Batch 2399/2400]' in i]
    # Dlosslist = [float(m.split('] [')[2][-8:].replace(']', '')) for m in tmp]
    Glosslist3 = [float(m.split('] [')[3][-8:].replace(']', '')) for m in tmp]

# with open('srgan_hrnet2.log', 'r', encoding='utf-8') as f:
#     data = f.read().split('\n')
#     tmp = [i for i in data if '[Batch 2399/2400]' in i]
#     # Dlosslist = [float(m.split('] [')[2][-8:].replace(']', '')) for m in tmp]
#     Glosslist4 = [float(m.split('] [')[3][-8:].replace(']', '')) for m in tmp]
#
# with open('srgan_hrnet3.log', 'r', encoding='utf-8') as f:
#     data = f.read().split('\n')
#     tmp = [i for i in data if '[Batch 2399/2400]' in i]
#     # Dlosslist = [float(m.split('] [')[2][-8:].replace(']', '')) for m in tmp]
#     Glosslist5 = [float(m.split('] [')[3][-8:].replace(']', '')) for m in tmp]

with open('srgandensenet.log', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    tmp = [i for i in data if '[Batch 4799/4800]' in i]
    # Dlosslist = [float(m.split('] [')[2][-8:].replace(']', '')) for m in tmp]
    Glosslist6 = [float(m.split('] [')[3][-8:].replace(']', '')) for m in tmp]

with open('srgan_hrnet_transformer.log', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    tmp = [i for i in data if '[Batch 3199/3200]' in i]
    # Dlosslist = [float(m.split('] [')[2][-8:].replace(']', '')) for m in tmp]
    Glosslist7 = [float(m.split('] [')[3][-8:].replace(']', '')) for m in tmp]

import matplotlib.pyplot as plt
# plt.subplot(2,1,1)
plt.title('Generator Loss')
# plt.plot(Glosslist1,c='orange',label='densenet32')
# plt.text(0,Glosslist1[0],(0,Glosslist1[0]),color='orange')
# plt.plot(Glosslist2,c='r',label = 'hrnet')
# plt.text(0,Glosslist2[0],(0,Glosslist2[0]),color='r')
# plt.plot(Glosslist3,c='g',label='densenet16')
# plt.text(0,Glosslist3[0],(0,Glosslist3[0]),color='g')
# plt.plot(Glosslist4,c='black',label='hrnet2')
# plt.text(0,Glosslist4[0],(0,Glosslist4[0]),color='black')
# plt.plot(Glosslist5,c='b',label='hrnet3')
# plt.text(0,Glosslist5[0],(0,Glosslist5[0]),color='b')
limit = 300
linewidth=0.5
plt.grid(b=None, which='major', axis='both')
plt.plot(Glosslist3[0:],c='g',label='ResNet16',linewidth=linewidth)
plt.text(0,Glosslist3[0],(0,Glosslist3[0]),color='g')
# plt.plot(Glosslist1[0:limit],c='orange',label='ResNet32',linewidth=linewidth)
# plt.text(0,Glosslist1[0],(0,Glosslist1[0]),color='orange')
# plt.plot(Glosslist5[0:limit],c='b',label='HRNet_3',linewidth=linewidth)
# plt.text(0,Glosslist5[0],(0,Glosslist5[0]),color='b')
# plt.plot(Glosslist4[0:limit],c='black',label='HRNet_2',linewidth=linewidth)
# plt.text(0,Glosslist4[0],(0,Glosslist4[0]),color='black')
plt.plot(Glosslist2[0:],c='r',label = 'HRNet_1',linewidth=1.5)
plt.text(0,Glosslist2[0],(0,Glosslist2[0]),color='r')
plt.plot(Glosslist6[0:],c='black',label = 'DenseNet16',linewidth=1.5)
plt.text(0,Glosslist6[0],(0,Glosslist2[0]),color='black')
plt.plot(Glosslist7[0:],c='yellow',label = 'srgan_hrnet_transformer',linewidth=1.5)
plt.text(0,Glosslist7[0],(0,Glosslist2[0]),color='yellow')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.subplots_adjust(wspace =1, hspace =1)#调整子图间距
plt.savefig('g2.svg')
print('picover')


print('resnet16',min(Glosslist3[0:]),Glosslist3.index(min(Glosslist3[0:])))
# print('densenet32',min(Glosslist1[90:limit]),Glosslist1.index(min(Glosslist1[90:limit])))
# print('hrnet3',min(Glosslist5[90:limit]),Glosslist5.index(min(Glosslist5[90:limit])))
# print('hrnet2',min(Glosslist4[90:limit]),Glosslist4.index(min(Glosslist4[90:limit])))
print('hrnet',min(Glosslist2[0:]),Glosslist2.index(min(Glosslist2[0:])))
print('densenet16',min(Glosslist6[0:]),Glosslist6.index(min(Glosslist6[0:])))
print('srgan_hrnet_transformer',min(Glosslist7[0:]),Glosslist7.index(min(Glosslist7[0:])))


