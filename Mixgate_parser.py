# import os
# import re
# from collections import defaultdict

# import os

# def rename_bench_files(directory):
#     """
#     将目录中所有 .bench.bench 文件重命名为 .bench
#     """
#     for filename in os.listdir(directory):
#         if filename.endswith(".bench.bench"):
#             old_path = os.path.join(directory, filename)
#             new_path = os.path.join(directory, filename[:-6])  # 去掉多余的 .bench
#             os.rename(old_path, new_path)
#             print(f"[INFO] Renamed: {old_path} -> {new_path}")

# def normalize_filename(filename):
#     """
#     移除多余的扩展名
#     """
#     if filename.endswith(".bench"):
#         filename = filename[:-6]
#     return filename


# def append_to_bench_file(directory, filename, content):
#     os.makedirs(directory, exist_ok=True)
#     file_path = os.path.join(directory, f"{filename}.bench")
#     if os.path.exists(file_path):
#         with open(file_path, 'a') as file:  
#             file.write(content)
#     else:
#         with open(file_path, 'w') as file:
#             file.write(content)

# def replace_nodes(file_content, dic_nodes):
#     pattern = r'\bn(\d+)|\bpo(\d+)|\ba(\d+)'
#     for i, line in enumerate(file_content):
#         if isinstance(line, str):
#             def replace(match):
#                 full_key = match.group(0)
#                 return str(dic_nodes.get(full_key, full_key))
#             file_content[i] = re.sub(pattern, replace, line)

# def count_dict_keys_in_list(dict_keys, string_list):
#     count_dict = defaultdict(int)
#     for key in dict_keys:
#         pattern = r'\b' + re.escape(key) + r'\b'
#         for string in string_list:
#             if re.fullmatch(pattern, string):
#                 count_dict[key] += 1
    
#     return dict(count_dict)

# def count_dict_keys_in_list(dict_keys, string_list):
#     count_dict = defaultdict(int)
#     for key in dict_keys:
#         pattern = r'\b' + re.escape(key) + r'\b'
#         for string in string_list:
#             if re.fullmatch(pattern, string):
#                 count_dict[key] += 1
    
#     return dict(count_dict)

# def extract_lines_from_files(directory, aimed_directory):
#     dic = {"0xe8": [0, 0, 0], "0x8e":[1, 0, 0], "0xb2":[0, 1, 0], "0xd4":[0, 0, 1]}

#     for filename in os.listdir(directory):
#         file_content = []
#         added_node_dic = []
#         added_nodes = 0
#         count_input = 1
#         no_nodes = 1
#         no_nodes = 1
#         count_output = 0
#         count_inner_index = 0
#         total_nodes = 0
#         inner_nodes = 0
#         inner_nodes = 0
#         dic_nodes = {}
#         nodes_appear = {}
#         # 规范化文件名
#         # normalized_filename = normalize_filename(filename)
#         file_path = os.path.join(directory, filename)
#         aimed_file_path = os.path.join(aimed_directory, filename)
#         # aimed_file_path = os.path.join(aimed_directory, f"{normalized_filename}.bench")
#         if os.path.isfile(file_path):  # 确保是文件
#             with open(file_path, 'r') as file:  # 打开文件
#                 dic_nodes["n0"] = 0
#                 lines = file.readlines()  # 读取所有行
#                 file_content.append("INPUT(n0)\n")
#                 for line in lines:
#                     if "LUT" not in line:
#                         file_content.append(line)
#                         if "INPUT" in line:
#                             input = line.split("(")[1].split(")")[0]
#                             dic_nodes[input] = no_nodes
#                             dic_nodes[input] = no_nodes
#                             count_input += 1
#                             no_nodes += 1
#                             no_nodes += 1
#                         elif "OUTPUT" in line:
#                             output = line.split("(")[1].split(")")[0]
#                             dic_nodes[output] = no_nodes
#                             dic_nodes[output] = no_nodes
#                             count_output += 1
#                             no_nodes += 1
#                             no_nodes += 1
#                         #append_to_bench_file(aimed_directory, filename, line)
#                     else:
#                         if "po" not in line:
#                             inner_node = line.split(" ")[0]
#                             #print("inner_node =", inner_node)
#                             dic_nodes[inner_node] = no_nodes
#                             no_nodes += 1
#                             dic_nodes[inner_node] = no_nodes
#                             no_nodes += 1
#                         if "0xe8" in line:
#                             content = line.split(" ")
#                             file_content.append(content[0] + " = " + "MAJ" + content[4] + " " + content[5] + " " + content[6])
#                         elif "0x8e" in line:
#                             content = line.split(" ")
#                             added_node_dic.append(f"a{added_nodes}")
#                             file_content.append(content[0] + " = " + "MAJ" + "(" + added_node_dic[added_nodes] + "," + " " + content[5] + " " + content[6])
#                             node = line.split("(")[1].split(",")[0]
#                             file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
#                             added_nodes += 1
                            
                            
#                         elif "0xb2" in line:
#                             content = line.split(" ")
#                             added_node_dic.append(f"a{added_nodes}")
#                             file_content.append(content[0] + " = " + "MAJ"  + content[4] + " " + added_node_dic[added_nodes] + "," + " " + content[6])
#                             node = line.split("(")[1].split(",")[1].split(" ")[1]
#                             file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
#                             added_nodes += 1
                            
                            
#                         elif "0xd4" in line:
#                             content = line.split(" ")
#                             added_node_dic.append(f"a{added_nodes}")
#                             file_content.append(content[0] + " = " + "MAJ"  + content[4] + " " + content[5]  + " " + added_node_dic[added_nodes] + ")" + '\n')
#                             node = line.split("(")[1].split(" ")[2].split(")")[0]
#                             file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
#                             added_nodes += 1
                            
                            
#                         elif "0x96" in line:
#                             content = line.split(" ")
#                             file_content.append(content[0] + " = " + "XOR" + content[4] + " " + content[5] + " " + content[6])
#                         elif "0x8" in line:
#                             content = line.split(" ")
#                             file_content.append(content[0] + " = " + "AND" + content[4] + " " + content[5])   
#                         elif "0x6" in line:
#                             content = line.split(" ")
#                             file_content.append(content[0] + " = " + "XOR" + content[4] + " " + content[5])  
#                         elif "0x4" in line:
#                             content = line.split(" ")
#                             added_node_dic.append(f"a{added_nodes}")
#                             file_content.append(content[0] + " = " + "AND"  + content[4] + " " + added_node_dic[added_nodes] + ")" + '\n')
#                             node = line.split("(")[1].split(" ")[1].split(")")[0]
#                             file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
#                             added_nodes += 1                                                                              
#                         # elif "0x2" in line and len(line.split(" ")) <= 5:
#                         #     content = line.split(" ")
#                         #     added_node_dic.append(f"a{added_nodes}")
#                         #     file_content.append(content[0] + " = " + "BUF"  + content[4])
#                         #     added_nodes += 1
#                         # elif "0x2" in line and len(line.split(" ")) <= 5:
#                         #     content = line.split(" ")
#                         #     added_node_dic.append(f"a{added_nodes}")
#                         #     file_content.append(content[0] + " = " + "BUF"  + content[4])
#                         #     added_nodes += 1
#                         elif "0x2" in line and len(line.split(" ")) >= 6:
#                             content = line.split(" ")
#                             #print(len(content))
#                             added_node_dic.append(f"a{added_nodes}")
#                             file_content.append(content[0] + " = " + "AND"  + "(" + added_node_dic[added_nodes] + "," + " " + content[5])
#                             node = line.split("(")[1].split(" ")[0].split(",")[0]
#                             file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
#                             added_nodes += 1           
                    
#                             added_nodes += 1           
                    
#                         elif "0x1" in line and len(line.split(" ")) <= 5:
#                             content = line.split(" ")
#                             added_node_dic.append(f"a{added_nodes}")
#                             file_content.append(content[0] + " = " + "NOT"  + content[4])
#                             added_nodes += 1


#                         elif "0x1" in line and len(line.split(" ")) >= 6:
#                             content = line.split(" ")
#                             added_node_dic.append(f"a{added_nodes}")
#                             added_nodes += 1  
#                             added_node_dic.append(f"a{added_nodes}")
#                             file_content.append(content[0] + " = " + "AND"  + "(" + added_node_dic[added_nodes - 1] + ", " + added_node_dic[added_nodes] + ")" + '\n')
#                             node_0 = line.split(" ")[4].split("(")[1].split(",")[0]
#                             node_1 = line.split(" ")[5].split(")")[0]
#                             file_content.append(added_node_dic[added_nodes - 1] + " = " + "NOT" + "(" + node_0 + ")" + '\n')
#                             file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node_1 + ")" + '\n')
#                             added_nodes += 1  

#                 # total_nodes = count_input + count_inner_index + 1
#                 # print("total_nodes =", no_nodes)
#                 # print("added_nodes =", added_nodes)

#                 # total_nodes = count_input + count_inner_index + 1
#                 # print("total_nodes =", no_nodes)
#                 # print("added_nodes =", added_nodes)
#                 for i in range(added_nodes):
#                     dic_nodes[f"a{i}"] = no_nodes
#                     no_nodes += 1
#                     dic_nodes[f"a{i}"] = no_nodes
#                     no_nodes += 1
#                 replace_nodes(file_content, dic_nodes)
#         if len(dic_nodes) > 1000:
#             continue                
#         with open(aimed_file_path, 'w') as file:
#             for item in file_content:
#                 if item != '0 = gnd\n':
#                     file.write(item)
           

# if __name__ == "__main__":
# # 使用示例
#     directory = '/home/wjx/MixGate/datasets/raw_bench/raw1_xag'
#     aimed_directory = '/home/wjx/MixGate/datasets/aimed_xag'

#     # 将 .bench.bench 文件重命名为 .bench
#     rename_bench_files(directory)

#     if not os.path.exists(aimed_directory):
#         os.makedirs(aimed_directory)
#     extracted_lines = extract_lines_from_files(directory, aimed_directory)




import os
import re
from collections import defaultdict

def append_to_bench_file(directory, filename, content):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{filename}.bench")
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:  
            file.write(content)
    else:
        with open(file_path, 'w') as file:
            file.write(content)

def replace_nodes(file_content, dic_nodes):
    pattern = r'\bn(\d+)|\bpo(\d+)|\ba(\d+)'
    for i, line in enumerate(file_content):
        if isinstance(line, str):
            def replace(match):
                full_key = match.group(0)
                return str(dic_nodes.get(full_key, full_key))
            file_content[i] = re.sub(pattern, replace, line)

def count_dict_keys_in_list(dict_keys, string_list):
    count_dict = defaultdict(int)
    for key in dict_keys:
        pattern = r'\b' + re.escape(key) + r'\b'
        for string in string_list:
            if re.fullmatch(pattern, string):
                count_dict[key] += 1
    
    return dict(count_dict)

def extract_lines_from_files(directory, aimed_directory):
    dic = {"0xe8": [0, 0, 0], "0x8e":[1, 0, 0], "0xb2":[0, 1, 0], "0xd4":[0, 0, 1]}

    for filename in os.listdir(directory):
        file_content = []
        added_node_dic = []
        added_nodes = 0
        count_input = 1
        no_nodes = 1
        count_output = 0
        count_inner_index = 0
        total_nodes = 0
        inner_nodes = 0
        dic_nodes = {}
        nodes_appear = {}
        file_path = os.path.join(directory, filename)
        aimed_file_path = os.path.join(aimed_directory, filename)
        if os.path.isfile(file_path):  # 确保是文件
            with open(file_path, 'r') as file:  # 打开文件
                #dic_nodes["n0"] = 0
                lines = file.readlines()  # 读取所有行
                #file_content.append("INPUT(n0)\n")
                for line in lines:
                    if "LUT" not in line:
                        file_content.append(line)
                        if "INPUT" in line:
                            input = line.split("(")[1].split(")")[0]
                            dic_nodes[input] = no_nodes
                            count_input += 1
                            no_nodes += 1
                        elif "OUTPUT" in line:
                            output = line.split("(")[1].split(")")[0]
                            dic_nodes[output] = no_nodes
                            count_output += 1
                            no_nodes += 1
                        #append_to_bench_file(aimed_directory, filename, line)
                    else:
                        if "po" not in line:
                            inner_node = line.split(" ")[0]
                            #print("inner_node =", inner_node)
                            dic_nodes[inner_node] = no_nodes
                            no_nodes += 1
                        if "0xe8" in line:
                            content = line.split(" ")
                            if "n0" in line:
                                file_content.append(content[0] + " = " + "AND" + "(" + content[5] + " " + content[6])
                            else:
                                file_content.append(content[0] + " = " + "MAJ" + content[4] + " " + content[5] + " " + content[6])
                        elif "0x8e" in line:
                            content = line.split(" ")
                            if "n0" in line:
                                file_content.append(content[0] + " = " + "OR" + "(" + content[5] + " " + content[6])
                            else:
                                added_node_dic.append(f"a{added_nodes}")
                                file_content.append(content[0] + " = " + "MAJ" + "(" + added_node_dic[added_nodes] + "," + " " + content[5] + " " + content[6])
                                node = line.split("(")[1].split(",")[0]
                                file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                                added_nodes += 1
                            
                        elif "0xb2" in line:
                            content = line.split(" ")
                            if "n0" in line:
                                added_node_dic.append(f"a{added_nodes}")
                                file_content.append(content[0] + " = " + "AND" + "(" + content[5] + " " + content[6])
                                node = line.split("(")[1].split(",")[1].split(" ")[1]
                                file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                                added_nodes += 1
                            else:
                                added_node_dic.append(f"a{added_nodes}")
                                file_content.append(content[0] + " = " + "MAJ"  + content[4] + " " + added_node_dic[added_nodes] + "," + " " + content[6])
                                node = line.split("(")[1].split(",")[1].split(" ")[1]
                                file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                                added_nodes += 1
                            
                        elif "0xd4" in line:
                            content = line.split(" ")
                            if "n0" in line:
                                added_node_dic.append(f"a{added_nodes}")
                                file_content.append(content[0] + " = " + "AND" + "(" + content[5] + " " + content[6])
                                node = line.split("(")[1].split(" ")[2].split(")")[0]
                                file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                                added_nodes += 1
                            else:
                                added_node_dic.append(f"a{added_nodes}")
                                file_content.append(content[0] + " = " + "MAJ"  + content[4] + " " + content[5]  + " " + added_node_dic[added_nodes] + ")" + '\n')
                                node = line.split("(")[1].split(" ")[2].split(")")[0]
                                file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                                added_nodes += 1
                            
                        elif "0x96" in line:
                            content = line.split(" ")
                            if "n0" in line:
                                file_content.append(content[0] + " = " + "XOR"  + content[4] + " " + content[5].split(",")[0] + ")" + '\n')
                            else:
                                file_content.append(content[0] + " = " + "XOR" + content[4] + " " + content[5] + " " + content[6])
                        elif "0x8" in line:
                            content = line.split(" ")
                            file_content.append(content[0] + " = " + "AND" + content[4] + " " + content[5])   
                        elif "0x6" in line:
                            content = line.split(" ")
                            file_content.append(content[0] + " = " + "XOR" + content[4] + " " + content[5])  
                        elif "0x4" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "AND"  + content[4] + " " + added_node_dic[added_nodes] + ")" + '\n')
                            node = line.split("(")[1].split(" ")[1].split(")")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1                                                                              
                        # elif "0x2" in line and len(line.split(" ")) <= 5:
                        #     content = line.split(" ")
                        #     added_node_dic.append(f"a{added_nodes}")
                        #     file_content.append(content[0] + " = " + "BUF"  + content[4])
                        #     added_nodes += 1
                        elif "0x2" in line and len(line.split(" ")) >= 6:
                            content = line.split(" ")
                            #print(len(content))
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "AND"  + "(" + added_node_dic[added_nodes] + "," + " " + content[5])
                            node = line.split("(")[1].split(" ")[0].split(",")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1           
                    
                        elif "0x1" in line and len(line.split(" ")) <= 5:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "NOT"  + content[4])
                            added_nodes += 1

                        elif "0x1" in line and len(line.split(" ")) >= 6:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            added_nodes += 1  
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "AND"  + "(" + added_node_dic[added_nodes - 1] + ", " + added_node_dic[added_nodes] + ")" + '\n')
                            node_0 = line.split(" ")[4].split("(")[1].split(",")[0]
                            node_1 = line.split(" ")[5].split(")")[0]
                            file_content.append(added_node_dic[added_nodes - 1] + " = " + "NOT" + "(" + node_0 + ")" + '\n')
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node_1 + ")" + '\n')
                            added_nodes += 1  

                # total_nodes = count_input + count_inner_index + 1
                # print("total_nodes =", no_nodes)
                # print("added_nodes =", added_nodes)
                for i in range(added_nodes):
                    dic_nodes[f"a{i}"] = no_nodes
                    no_nodes += 1
                # print("dic_nodes =", dic_nodes)
                # print("file_content =", file_content)
                # dic = count_dict_keys_in_list(dic_nodes, file_content)
                # print("dic =", dic)
                replace_nodes(file_content, dic_nodes)
                # print("file_content =", file_content)
        if len(dic_nodes) > 5000:
            continue                
        with open(aimed_file_path, 'w') as file:
            for item in file_content:
                if item != 'n0 = gnd\n':
                    file.write(item)




                            

if __name__ == "__main__":
# 使用示例
    directory = '/home/jwt/ext_xmg'
    aimed_directory = '/home/jwt/aimed_xmg'
    if not os.path.exists(aimed_directory):
        os.makedirs(aimed_directory)
    extracted_lines = extract_lines_from_files(directory, aimed_directory)

