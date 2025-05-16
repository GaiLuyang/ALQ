from datetime import datetime
import pytz
def create_file(task  = None):
    data_file_name = ['mnist', 'cifar10']
    if task.opt == 'ALQ':
        opt = '_ALQ_'

    if task.fl == 'AVG':
        fl = '_AVG'
    elif task.fl == 'ADMM':
        fl = '_ADMM'


    if task.exper == 1: 
        file_name = '1_' + data_file_name[task.choose] + fl + opt
    elif task.exper == 2:
        file_name = '2_' + data_file_name[task.choose] + fl + opt 
    elif task.exper == 3:
        file_name = '3_'+data_file_name[task.choose] + fl + opt + str(task.ex) + '_'
    elif task.exper == 4:
        file_name = '4_' + data_file_name[task.choose] + fl + opt + str(task.noniid_level) + '_'      
    elif task.exper == 5:
        file_name = '5_' + data_file_name[task.choose] + fl + opt + str(task.client_need) + '_'      

    file_name = file_name + task.exp_num
    print('Create file: ' , file_name)
    return file_name
    
def save_result(LOSS_test, Acc, model_bound, bits, energies, up_bits, cum_cost, cal_cost, total_cost, task):
    if task.machine == 'server':
        way = './ALQ-mnist/result/exp' + str(task.exper) + '/'
    elif task.machine == 'notebook':
        way = './result/exp' + str(task.exper) + '/'
    
    result_file_name = way + task.file_name.split('_')[1] + '/' + task.fl + '/' + task.file_name + '.txt'      

    with open(result_file_name, 'w', encoding='utf-8') as file:
            file.write('Loss:')
            file.write("\n")
            file.write(str(LOSS_test) + '\n')

            file.write('Acc:')
            file.write("\n")
            file.write(str(Acc) + '\n')

            file.write('Bits:')
            file.write("\n")
            file.write(str(bits) + '\n')

            file.write('Up_Bits:')
            file.write("\n")
            file.write(str(up_bits) + '\n')

            file.write('Cum_Cost:')
            file.write("\n")
            file.write(str(cum_cost) + '\n')

            file.write('Cal_Cost:')
            file.write("\n")
            file.write(str(cal_cost) + '\n')

            file.write('Total_Cost:')
            file.write("\n")
            file.write(str(total_cost) + '\n')

            file.write('Energies:')
            file.write("\n")
            energies_list = [0]
            for i in range(1,len(energies)):
                energies_list.append(energies[i].item())
            file.write(str(energies_list) + '\n')

            file.write('Eps:')
            file.write("\n")
            file.write(str(task.eps_sum) + '\n')

            file.write('Time:')
            file.write("\n")
            beijing_tz = pytz.timezone('Asia/Shanghai')
            beijing_now = datetime.now(beijing_tz)
            formatted_beijing_now = beijing_now.strftime("%Y-%m-%d %H:%M:%S")
            file.write(str(formatted_beijing_now) + '\n')
            
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write('Task parameters:')  
            file.write("\n")
            attributes = vars(task)
            for key, value in attributes.items():
                file.write(f"{key}: {value}\n")
