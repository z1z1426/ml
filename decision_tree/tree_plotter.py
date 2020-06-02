# 代码详解：https://blog.csdn.net/qq_44124157/article/details/88427339
import matplotlib.pyplot as plt


# decision_node:决策节点
# leaf_node:叶子节点
# 定义文本框和箭头格式
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# 这个是用来一注释形式绘制节点和箭头线，可以不用管
def plot_node(node_txt, center_pt, parent_pt, node_type):
    # 提供辅助标注的函数
    # 参数;
    # node_txt:标注文本
    # xy:箭头坐标
    # xytext:文本坐标
    # arrowprops={facecolor= '颜色',shrink = '数字' <1  收缩箭头}
    # axes fraction:轴分数
    # xycoords,textcoords指定坐标系
    # | 参数 | 坐标系 |
    # | 'figure points' | 距离图形左下角的点数量 |
    # | 'figure pixels' | 距离图形左下角的像素数量 |
    # | 'figure fraction' | 0,0 是图形左下角，1,1 是右上角 |
    # | 'axes points' | 距离轴域左下角的点数量 |
    # | 'axes pixels' | 距离轴域左下角的像素数量 |
    # | 'axes fraction' | 0,0 是轴域左下角，1,1 是右上角 |
    # | 'data' | 使用轴域数据坐标系 |
    # 以文本坐标指定center_pt
    # ha = "center"
    # 在水平方向上，方框的中心在为center_pt
    # va = "center"
    # 在垂直方向上，方框的中心在为center_pt
    # bbox:
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def get_num_leafs(mytree):
    """
    获取叶子节点的数目
    :param my_tree:决策树字典
    :return:
    """
    # 定义叶子节点
    num_leafs = 0
    # 取出划分数据集的第一个特征的字符串
    # (决策树树字典的每个分支的根节点的key)
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        # 判断key对应的数据结构是否是字典
        # 字典---决策节点---继续划分数据
        # 非字典---叶子节点
        # if type(second_dict[key]).__name__ == 'dict':
        if isinstance(second_dict[key], dict):
            # 若是决策节点,则以递归的方式获取当前决策节点拥有的叶子节点数
            num_leafs += get_num_leafs(second_dict[key])
        else:
            # 是叶子节点,则叶子节点计数器加1
            num_leafs += 1
    # 遍历完当前树层数所有的节点,筛选出所有的叶子节点数并返回
    return num_leafs


def get_tree_depth(my_tree):
    """
    获取树的层数
    :param my_tree:决策树字典
    :return:
    """
    # 定义当前树最大的深度
    max_depth = 0
    # 取出划分数据集的第一个特征的字符串
    # (决策树树字典的每个分支的根节点的key)
    first_str = list(my_tree.keys())[0]
    # 取出划分的数据集(字典形式表示)
    second_dict = my_tree[first_str]
    # 遍历数据及所有的key
    for key in second_dict.keys():
        # 判断key对应的数据结构是否是字典
        # 字典---决策节点---继续划分数据
        # 非字典---叶子节点
        # if type(second_dict[key]).__name__ == 'dict':
        if isinstance(second_dict[key], dict):
            # 若是决策节点,则以递归的方式继续搜索决策树分支的深度
            # 并将当前深度加1
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            # 若是叶子节点,则当前划分的数据集深度为1
            this_depth = 1
        # 若当前深度大于最大深度则刷新最大深度的值
        # 遍历完分支节点则获取到最大的深度值
        if this_depth > max_depth:
            max_depth = this_depth
    # 返回最大深度值
    return max_depth


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                     ]
    return list_of_trees[i]


# 这个是用来绘制线上的标注，简单
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    在父节点与子节点的箭头中间绘制txt_str
    :param cntr_pt: 子节点坐标
    :param parent_pt: 父节点坐标
    :param txt_str: 注释文本
    :return:
    """
    # 获取注释文本的x坐标
    xmid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    # 获取注释文本的y坐标
    ymid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    # 调用ax1图像的方法text()---在图像对象的给定坐标上呈现文本txt_str
    create_plot.ax1.text(xmid, ymid, txt_string)


def plot_tree(mytree, parent_pt, node_txt):
    num_leafs = get_num_leafs(mytree)
    # depth = get_tree_depth(mytree)
    first_str = list(mytree.keys())[0]
    # 子节点的x坐标:计算原理
    # 目标:使用切割法保证当前决策树节点的坐标在其占有的区域的中间(切割掉不属于决策节点的叶子节点的区域)
    # x,y轴取值范围 -- 0至1
    # 解释:(以深度优先遍历树型节点为基础---保证叶子节点会按照坐标从左往右依次绘制)
    #  叶子节点坐标确定:
    #  我们是按照叶子节点的数目均分画布的宽度,并且需要让叶子节点处于占有区间的中间位置
    #  那么叶子节点的坐标必然是占有宽度加上一个叶子节点的坐标作为标志坐标
    #  公式:new_point_x = prev_point_x + width
    #     plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
    #     (第一个叶子节点必须向左偏移其宽度一半,才能处于中间,所以初始plot_tree.x_off = -(total_width / total_num) / 2
    #     ======>  plot_tree.x_off = -0.5 / plot_tree.total_w)
    #  决策节点坐标确定:
    #  当前决策节点并不占有其不包含的叶子节点们占有的画布区域,计算决策节点占有宽度:
    #  (num / total_num) * width ====> ( float(num_leafs) / plot_tree.total_w ) * 1
    #  决策节点的坐标值就是应该等于其不包含的叶子节点占有的画布宽度作为起 始标志坐标 + 决策节点占有宽度的一半(就处于其占有区域的中间位置)
    #  决策节点的一半宽度:
    #  ( float(num_leafs) / plot_tree.total_w ) * 1 / 2.0 ===> ( float(num_leafs) / plot_tree.total_w ) / 2.0
    #  不包含的叶子节点的宽度:
    #  最近一个叶子节点的坐标 + 向左偏移的半个单叶子节点宽度() = 非包含所有叶子节点宽度
    #  plot_tree.x_off + 0.5 / plot_tree.total_w
    #  最终决策节点坐标公式:
    #  plot_tree.x_off + 0.5 / plot_tree.total_w + ( float(num_leafs) / plot_tree.total_w ) / 2.0
    # 变形====> plot_tree.x_off + 0.5 * 2 / plot_tree.total_w / 2.0 + float(num_leafs) / plot_tree.total_w  / 2.0
    # 合并同类项====> plot_tree.x_off + (0.5 * 2  + float(num_leafs) )/ plot_tree.total_w  / 2.0
    # 合并同类项====> plot_tree.x_off + (1.0  + float(num_leafs) ) / 2.0 / plot_tree.total_w
    # 原始根节点也可以通过此公式计算,并且计算的时候,父节点和子节点的坐标重合,此时
    # 就不会出现绘制的箭头
    cntr_pt = (plot_tree.x0ff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.y0ff)
    # 把分支上的键值,画在所属箭头中间
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    # 绘制父节点(注意:所有的父节点必然是决策节点,因为只有决策节点才需要划分数据集,即其数据类型是一个字典)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    # 取出当前父节点的子集字典
    second_dict = mytree[first_str]
    # 进入下一层节点的绘制,更新本层节点的y坐标(减去一层的高度)
    plot_tree.y0ff = plot_tree.y0ff - 1.0 / plot_tree.totalD
    # 遍历所以子节点的key
    for key in second_dict.keys():
        # 判断节点是否是决策节点
        if isinstance(second_dict[key], dict):
            # 若是决策节点,以递归的方式继续展开决策树,并且绘制
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            # 若是叶子节点,直接绘制本节点,并且本节点绘制结束,返回父节点高度
            # 确定本节点的x坐标
            # 原理:本函数的方式是以深度优先的方式遍历树型节点,也就是绘制的顺正好是
            # 从叶子节点中x坐标最小的开始绘制,依次下一个,知道绘制到最后一个
            # 叶子节点的坐标计算:plot_tree.x_off的值在最原始是第一个叶子坐标向左的偏移量
            # 1.0 / plot_tree.total_w表示每个叶子节点的占有宽度,第一个偏移量是-0.5 / plot_tree.total_w(宽度一半)
            # 即第一个叶子节点坐标会在其占有区域的中间,然后使用plot_tree.x_off保存第一个叶子节点的坐标
            # 第一个叶子节点等于它的宽度加上第一个叶子节点坐标作为偏移量,以此类推
            plot_tree.x0ff = plot_tree.x0ff + 1.0 / plot_tree.totalW
            # 绘制叶子节点
            plot_node(second_dict[key], (plot_tree.x0ff, plot_tree.y0ff), cntr_pt, leaf_node)
            # 为叶子节点绘制键值标注
            plot_mid_text((plot_tree.x0ff, plot_tree.y0ff), cntr_pt, str(key))
    # 返回上一层节点的绘制,更新本层节点的y坐标(加上一层的高度)
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

