import matplotlib.pyplot as plt


decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# 这个是用来一注释形式绘制节点和箭头线，可以不用管
def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)

    plt.show()


def get_num_leafs(mytree):
    num_leafs = 0
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(mytree):
    max_depth = 0
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                     ]
    return list_of_trees[i]


# 这个是用来绘制线上的标注，简单
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    xmid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    ymid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(xmid, ymid, txt_string)


def plot_tree(mytree, parent_pt, node_txt):
    num_leafs = get_num_leafs(mytree)
    depth = get_tree_depth(mytree)
    first_str = list(mytree.keys())[0]
    cntr_pt = (plot_tree.x0ff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.y0ff)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = mytree[first_str]
    plot_tree.y0ff = plot_tree.y0ff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x0ff = plot_tree.x0ff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.x0ff, plot_tree.y0ff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x0ff, plot_tree.y0ff), cntr_pt, str(key))
    plot_tree.y0ff = plot_tree.y0ff + 1.0 / plot_tree.totalD


def create_plot(intree):
    """
    in_tree:字典形式的决策树结构
    """
    # figure:图形,数字,创建一个图形实例
    # num = None:1-N,如果不提供,则增加figure的计数数值,如果提供,则在已存在中寻找,未找到创建,如果是字符串,则设置成窗口名
    # figsize = None:以英寸为单位的宽高，缺省值为figure.figsize
    # dpi = None:图形分辨率，缺省值为figure.dpi
    # facecolor = None,背景色
    # edgecolor = None,边框颜色
    # frameon = True,默认值True为绘制边框，如果为False则不绘制边框
    # FigureClass = Figure,matplotlib.figure.Figure派生类，从派生类创建figure实例
    # clear = False,重建figure实例
    fig = plt.figure(1, facecolor='white')
    # 清除图形
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # subplot是将多个图画到一个平面上的工具
    # 参数;
    # m:
    # n:
    # m和n代表在一个图像窗口中显示m行n列个图像
    # p:p表示图所在的位置，p=1表示从左到右从上到下的第一个位置
    #  frameon=False),添加没有框架的子图
    # 定义一个全局绘图区:create_plot.ax1
    # 给函数绑定属性,则函数全局可以使用这些属性
    # ax1---全局的图像对象
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(intree))
    plot_tree.totalD = float(get_tree_depth(intree))
    # (1 / plot_tree.total_w)---叶子节点能占有的宽度
    # 为了让叶子节点在其拥有宽度的区域居中
    # 故向左移动其所占有宽度一般
    # x_0ff:居中对齐的节点坐标偏移量
    plot_tree.x0ff, plot_tree.y0ff = -0.5 / plot_tree.totalW, 1.0
    # 绘制决策树--头像是1x1,所以(0.5,1.0)---根节点的坐标--横轴居中
    plot_tree(intree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    mytree = retrieve_tree(1)
    create_plot(mytree)