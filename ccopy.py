import javalang
from javalang.ast import Node
from tqdm import tqdm
import torch
import numpy as np
from Node import ASTNode,SingleNode
import copy




path_dict,node_dict,p_n_dict = dict(),dict(),dict()
pathID=-1
ID=-1


def say(self, content):
    print(content)


def java_get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def java_get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def java_get_node_list(ast):
    node_list, src, dst, deep_label_list, root = [], [], [], [], []
    deep_label = 0
    global ID
    ID += 1
    root.append(ID)
    path_list, pathID_list = [], []
    java_get_node(ast, node_list, ID, src, dst, deep_label, deep_label_list, path_list, pathID_list)

    return node_list, deep_label_list, src, dst, root


def java_get_node(node, node_list, id, src, dst, deep_label, deep_label_list, path_list, pathID_list):
    global ID, path_dict, node_dict, p_n_dict
    token, children = java_get_token(node), java_get_children(node)
    path_list.append(token + str(id))
    pathID_list.append(id)
    node_content = ''
    if isinstance(node, Node):
        node_content = node_content + str(node.position)
    if len(children) == 0:
        node_content = str(token)
        global pathID
        pathID += 1
        path_dict[pathID] = copy.deepcopy(path_list)
        p_n_dict[pathID] = copy.deepcopy(pathID_list)


    else:
        node_content = str(token)

    id = ID
    node_list.append(node_content)
    print(id, node_content)
    node_dict[id] = node_content
    deep_label_list.append(deep_label)

    for child in children:
        if java_get_token(child) == '':
            continue
        ID += 1
        src.append(id)
        dst.append(ID)
        java_get_node(child, node_list, ID, src, dst, deep_label + 1, deep_label_list, path_list, pathID_list)
        path_list.pop()
        pathID_list.pop()


id, node_list, deep_label_list, src, dst, root, file_label = [], [], [], [], [], [], []


def get_java_ast(code):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

with open("E:/00-Adata/code.txt", "r") as f:  # 打开文件
    data = f.read()  # 读取文
    print(data)
    ast = get_java_ast(data)
    node_list_single, deep_label_list_single, src_single, dst_single, root_single = java_get_node_list(ast)
    print(len(deep_label_list_single))

    # for key, value in path_dict.items():
    #     print('key: ', key, 'value: ', value)
    #
    # for key, value in p_n_dict.items():
    #     print('key: ', key, 'value: ', value)



'''

'''

