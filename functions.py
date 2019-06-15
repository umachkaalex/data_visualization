from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib
import pandas as pd
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button as pltButton
from matplotlib.lines import Line2D
from matplotlib.axes import Axes


def visualize_data_graph(input_data_file, target_column_visualize):
    all_columns_list = input_data_file.columns.values.tolist()
    output_column = target_column_visualize
    all_columns_list.remove(output_column)
    outputs_unique = list(set(input_data_file[output_column].astype(str).unique()))
    all_colors = ['#FF1111', '#11FF11', '#1111FF', '#111111', '#FFFF11',
                  '#FF660D', '#CD19FF', '#19AFFF', '#00A0FF', '#E80C1A',
                  '#8F0078', '#008C65', '#5D9100', '#99690B', '#3C8E8F',
                  '#FF7070', '#70FF70', '#7070FF', '#707070', '#FFFE70']
    colors = all_colors[:len(outputs_unique)]
    colors_table = pd.DataFrame({str(output_column): outputs_unique, 'Colors': colors})

    combinations = []
    column_id = []
    comb_ind = 0
    for l in range(len(all_columns_list)-1):
        if l == 0:
            column_id.append(l)
        else:
            column_id.append(comb_ind)
        for i in range(1, len(all_columns_list) - l):
            combinations.append([l, l + i])
            comb_ind += 1


    data_matrix = input_data_file[all_columns_list].astype(float).values
    output_column_values = input_data_file[output_column].astype(str).values
    data_matrix = np.c_[data_matrix, np.matrix(output_column_values).T]

    fig = plt.figure(num='Visualize Columns Pairs vs Output')

    def calc_cur_combination_coord(i):
        first = combinations[i][0]
        second = combinations[i][1]
        cur_columns = [all_columns_list[first], all_columns_list[second],
                       output_column, 'Label']
        label_column = np.core.defchararray.add(np.asarray(data_matrix[:, first]).reshape(-1).astype(str),
                                                np.asarray(data_matrix[:, second]).reshape(-1).astype(str))

        cur_columns_matrix = data_matrix[:, [first, second, len(np.asarray(data_matrix[:1, :]).reshape(-1)) - 1]]
        cur_columns_matrix = np.c_[cur_columns_matrix, np.matrix(label_column).T]
        cur_columns_df = pd.DataFrame(cur_columns_matrix, columns=cur_columns)
        cur_columns_pvt_1 = cur_columns_df.groupby(cur_columns).size().reset_index(name='Outputs_XY')
        cur_columns_pvt_2 = cur_columns_df.groupby(cur_columns[-1]).size().reset_index(name='XY')
        cur_columns_pvt = pd.merge(cur_columns_pvt_1, cur_columns_pvt_2, on=['Label'])
        cur_columns_pvt = pd.merge(cur_columns_pvt, colors_table, on=[output_column])
        cur_columns_pvt['Scale'] = cur_columns_pvt['Outputs_XY'] / cur_columns_pvt['XY'].max() * 1000
        label_unique = cur_columns_pvt['Scale'].unique()
        xy_counts = cur_columns_pvt['Outputs_XY'].unique()
        label_unique = sorted(label_unique, key=int, reverse=True)
        xy_counts = sorted(xy_counts, key=int, reverse=True)

        cur_label_scale_values_list = []
        cur_label_colors_values_list = []
        cur_label_X_values_list = []
        cur_label_Y_values_list = []

        for i in range(len(label_unique)):
            cur_label_values = cur_columns_pvt.loc[cur_columns_pvt['Scale'] == label_unique[i]]
            cur_label_scale_values_list.append(cur_label_values['Scale'].tolist())
            cur_label_colors_values_list.append(cur_label_values['Colors'].tolist())
            cur_label_X_values_list.append(cur_label_values[all_columns_list[first]].tolist())
            cur_label_Y_values_list.append(cur_label_values[all_columns_list[second]].tolist())

        return [cur_label_X_values_list, cur_label_Y_values_list,
                cur_label_colors_values_list, cur_label_scale_values_list, xy_counts,
                first, second, label_unique]

    cur_scatter_data = calc_cur_combination_coord(0)

    plt.subplots_adjust(left=0.1, right=0.90)

    def new_scatters(cur_scatter_data):
        global fig_legend
        for l in range(len(cur_scatter_data[7])):
            ax.scatter(cur_scatter_data[0][l], cur_scatter_data[1][l],
                           c=cur_scatter_data[2][l], s=cur_scatter_data[3][l], alpha=0.7, edgecolors=None)

        legend_elements = [Line2D([0], [0], marker='o', color='grey', label=str(cur_scatter_data[4][i]),
                                  markerfacecolor='grey', markersize=cur_scatter_data[7][i] / 1000 * 30) for i in
                           range(len(cur_scatter_data[7]))]
        fig_legend = fig.legend(handles=legend_elements)

        outputs_elements = [Line2D([0], [0], marker='o', color=colors[i], label=outputs_unique[i],
                            markerfacecolor=colors[i], markersize=10) for i in range(len(outputs_unique))]
        ax.legend(handles=outputs_elements, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=5, mode="expand", borderaxespad=0.)

        ax.set_xlabel(str(all_columns_list[cur_scatter_data[5]]))
        ax.set_ylabel(str(all_columns_list[cur_scatter_data[6]]))

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    ax = fig.add_subplot(111)
    plt.suptitle('Target Column: ' + str(output_column))
    new_scatters(cur_scatter_data)

    class Index(object):
        ind = 0
        ind_col = 0
        def next(self, event):
            if self.ind < len(combinations)-1:
                self.ind += 1
                self.ind_col = np.digitize(self.ind, column_id) - 1
            else:
                self.ind = 0
                self.ind_col = np.digitize(self.ind, column_id) - 1
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            fig_legend.remove()
            Axes.clear(ax)
            new_scatters(cur_scatter_data)
            plt.draw()

        def prev(self, event):
            if self.ind > 0:
                self.ind -= 1
                self.ind_col = np.digitize(self.ind, column_id) - 1
            else:
                self.ind = len(combinations)-1
                self.ind_col = np.digitize(self.ind, column_id) - 1
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            fig_legend.remove()
            Axes.clear(ax)
            new_scatters(cur_scatter_data)
            plt.draw()

        def next_col(self, event):
            if self.ind_col + 1 <= len(column_id)-1:
                self.ind_col += 1
                self.ind = column_id[self.ind_col]
            else:
                self.ind_col = 0
                self.ind = column_id[self.ind_col]
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            fig_legend.remove()
            Axes.clear(ax)
            new_scatters(cur_scatter_data)
            plt.draw()

        def prev_col(self, event):
            if self.ind_col > 0:
                self.ind_col -= 1
                self.ind = column_id[self.ind_col]
            else:
                self.ind_col = len(column_id)-1
                self.ind = column_id[self.ind_col]

            cur_scatter_data = calc_cur_combination_coord(self.ind)
            fig_legend.remove()
            Axes.clear(ax)
            new_scatters(cur_scatter_data)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.01, 0.03, 0.08, 0.05])
    axnext = plt.axes([0.91, 0.03, 0.08, 0.05])
    bnext = pltButton(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = pltButton(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    ax_col_prev = plt.axes([0.11, 0.03, 0.08, 0.05])
    ax_col_next = plt.axes([0.81, 0.03, 0.08, 0.05])
    b_col_next = pltButton(ax_col_next, 'Next Col')
    b_col_next.on_clicked(callback.next_col)
    b_col_prev = pltButton(ax_col_prev, 'Previous Col')
    b_col_prev.on_clicked(callback.prev_col)

    plt.show()


def visualize3d_data_graph(input_data_file, target_column_visualize, main_frame_width=1024, main_frame_height=768):
    from traits.api import HasTraits, Instance, Button, Range, \
        on_trait_change
    from traitsui.api import View, Item, HGroup, VGroup, Group

    from mayavi import mlab
    from mayavi.core.ui.api import MlabSceneModel, SceneEditor
    all_columns_list = input_data_file.columns.values.tolist()

    output_column = target_column_visualize
    all_columns_list.remove(output_column)
    outputs_unique = list(set(input_data_file[output_column].astype(str).unique()))
    all_colors = [(1, 0.04, 0.04), (0.04, 1, 0.04), (0.04, 0.04, 1), (0.04, 0.04, 0.04), (1, 1, 0.04),
                  (1, 0.4, 0.05), '#CD19FF', '#19AFFF', '#00A0FF', '#E80C1A',
                  '#8F0078', '#008C65', '#5D9100', '#99690B', '#3C8E8F',
                  '#FF7070', '#70FF70', '#7070FF', '#707070', '#FFFE70']
    colors = all_colors[:len(outputs_unique)]
    points_dict = dict()
    scallars_len_list = []
    labels = []
    combinations = []
    next_column_start_combintaions_index = []
    comb_ind = 0
    column_ind = 0
    for l in range(len(all_columns_list)):
        for i in range(1, len(all_columns_list) - l):
            for z in range(1, len(all_columns_list) - i - l):
                combinations.append([l, l + i, l + i + z])
                if not next_column_start_combintaions_index:
                    next_column_start_combintaions_index.append(comb_ind)
                elif l == column_ind:
                    comb_ind += 1
                else:
                    comb_ind += 1
                    next_column_start_combintaions_index.append(comb_ind)
                    column_ind += 1

    data_matrix = input_data_file[all_columns_list].values
    output_column_values = input_data_file[output_column].astype(str).values
    data_matrix = np.c_[data_matrix, np.matrix(output_column_values).T]
    colors_table = pd.DataFrame({str(output_column): outputs_unique, 'Colors': colors})

    def calc_cur_combination_coord(i):
        first = combinations[i][0]
        second = combinations[i][1]
        third = combinations[i][2]
        cur_columns = [all_columns_list[first], all_columns_list[second], all_columns_list[third],
                       output_column, 'Label']

        label_column = np.core.defchararray.add(np.asarray(data_matrix[:, first]).reshape(-1).astype(str),
                                                np.asarray(data_matrix[:, second]).reshape(-1).astype(str))

        label_column = np.core.defchararray.add(label_column,
                                                np.asarray(data_matrix[:, third]).reshape(-1).astype(str))

        cur_columns_matrix = data_matrix[:, [first, second, third, len(np.asarray(data_matrix[:1, :]).reshape(-1)) - 1]]
        cur_columns_matrix = np.c_[cur_columns_matrix, np.matrix(label_column).T]
        cur_columns_df = pd.DataFrame(cur_columns_matrix, columns=cur_columns)
        cur_columns_pvt_1 = cur_columns_df.groupby(cur_columns).size().reset_index(name='Outputs_XYZ')
        cur_columns_pvt_2 = cur_columns_df.groupby(cur_columns[-1]).size().reset_index(name='XYZ')
        cur_columns_pvt = pd.merge(cur_columns_pvt_1, cur_columns_pvt_2, on=['Label'])
        cur_columns_pvt = pd.merge(cur_columns_pvt, colors_table, on=[output_column])
        cur_columns_pvt['Scale'] = cur_columns_pvt['Outputs_XYZ'] / cur_columns_pvt['XYZ'].max()
        label_unique = cur_columns_pvt['Colors'].astype(str).unique()

        cur_label_scale_values_list = []
        cur_label_colors_values_list = []
        cur_label_X_values_list = []
        cur_label_Y_values_list = []
        cur_label_Z_values_list = []

        for i in range(len(label_unique)):
            cur_label_values = cur_columns_pvt.loc[cur_columns_pvt['Colors'].astype(str) == str(label_unique[i])]
            cur_label_scale_values_list.append(cur_label_values['Scale'].tolist())
            cur_label_colors_values_list.append(cur_label_values['Colors'].unique())
            cur_label_X_values_list.append(cur_label_values[all_columns_list[first]].tolist())
            cur_label_Y_values_list.append(cur_label_values[all_columns_list[second]].tolist())
            cur_label_Z_values_list.append(cur_label_values[all_columns_list[third]].tolist())

        return [cur_label_X_values_list, cur_label_Y_values_list, cur_label_Z_values_list,
                cur_columns, cur_label_colors_values_list, cur_label_scale_values_list,
                label_unique]

    class MyDialog(HasTraits):
        ind = -1
        column_ind = -1
        scene1 = Instance(MlabSceneModel, ())
        cur_scale = 1
        cur_opacity = 50
        scale_range = Range(1, 200, cur_scale) #mode='spinner')
        opacity_range = Range(0, 100, cur_opacity)  # mode='spinner')
        button1 = Button('Next')
        button2 = Button('Previous')
        button3 = Button('Start Next Column')
        button4 = Button('Start Prev Column')

        @on_trait_change('button1')
        def redraw_scene1(self):
            if self.ind < len(combinations) - 1:
                self.ind += 1
            else:
                self.ind = 0
            del labels[:], scallars_len_list[:]
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            labels.append(cur_scatter_data[6])
            scallars_len_list.append(cur_scatter_data[5])
            self.redraw_scene(self.scene1, cur_scatter_data)


        @on_trait_change('button2')
        def redraw_scene2(self):
            if self.ind > 0:
                self.ind -= 1
            else:
                self.ind = len(combinations)-1
            del labels[:], scallars_len_list[:]
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            labels.append(cur_scatter_data[6])
            scallars_len_list.append(cur_scatter_data[5])
            self.redraw_scene(self.scene1, cur_scatter_data)

        @on_trait_change('button3')
        def redraw_scene3(self):
            if self.column_ind < len(next_column_start_combintaions_index) - 1:
                self.column_ind += 1
            else:
                self.column_ind = 0
            self.ind = next_column_start_combintaions_index[self.column_ind]
            del labels[:], scallars_len_list[:]
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            labels.append(cur_scatter_data[6])
            scallars_len_list.append(cur_scatter_data[5])
            self.redraw_scene(self.scene1, cur_scatter_data)

        @on_trait_change('button4')
        def redraw_scene4(self):
            if self.column_ind > 0:
                self.column_ind -= 1
            else:
                self.column_ind = len(next_column_start_combintaions_index) - 1
            self.ind = next_column_start_combintaions_index[self.column_ind]
            del labels[:], scallars_len_list[:]
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            labels.append(cur_scatter_data[6])
            scallars_len_list.append(cur_scatter_data[5])
            self.redraw_scene(self.scene1, cur_scatter_data)

        def redraw_scene(self, scene, cur_scatter_data):
            # Notice how each mlab call points explicitely to the figure it
            # applies to.
            mlab.clf(figure=scene.mayavi_scene)
            x = cur_scatter_data[0]
            y = cur_scatter_data[1]
            z = cur_scatter_data[2]
            xlabel_text = cur_scatter_data[3][0]
            ylabel_text = cur_scatter_data[3][1]
            zlabel_text = cur_scatter_data[3][2]
            cur_colors = cur_scatter_data[4]
            cur_scales = [np.array(x) for x in cur_scatter_data[5]]
            label_unique = cur_scatter_data[6]
            points_dict.clear()
            for ii in range(len(label_unique)):
                points_dict[label_unique[ii]] = mlab.points3d(x[ii], y[ii], z[ii], cur_scales[ii]*self.cur_scale, color=cur_colors[ii][0],
                                                                scale_factor=0.01, opacity=self.cur_opacity/100)
            for ii in range(len(outputs_unique)):
                mlab.text(0.02, 1 - 0.035 * (ii + 1), '-' + str(outputs_unique[ii]), width=0.06, color=all_colors[ii])

            cur_axex = mlab.axes(xlabel=xlabel_text, ylabel=ylabel_text, zlabel=zlabel_text,
                                 extent=[min(sum(x, [])), max(sum(x, [])), min(sum(y, [])),
                                         max(sum(y, [])), min(sum(z, [])), max(sum(z, []))])
            cur_axex.axes.font_factor = 0.8

        @on_trait_change('scale_range, scene.activated')
        def update_range(self):
            label_unique = labels[0]
            self.cur_scale = self.scale_range
            cur_scallars_len_list = [np.array(x) for x in scallars_len_list[0]]
            for ii in range(len(label_unique)):
                points_dict[label_unique[ii]].mlab_source.set(
                    scalars=cur_scallars_len_list [ii]*self.cur_scale)

        @on_trait_change('opacity_range, scene.activated')
        def update_opacity(self):
            label_unique = labels[0]
            self.cur_opacity = self.opacity_range
            for ii in range(len(label_unique)):
                points_dict[label_unique[ii]].mlab_source.set(opacity=self.cur_opacity/100)

        # The layout of the dialog created
        view = View(
                    VGroup(Item('scene1',
                                editor=SceneEditor(), height=main_frame_height,
                                width=main_frame_width),
                           HGroup(HGroup('button1', 'button2', show_labels=False),
                                  HGroup('scale_range','opacity_range'),
                                  HGroup('button3', 'button4', show_labels=False)),
                           show_labels=False),
                    resizable = True
                    )

    m = MyDialog()
    m.configure_traits()


def visualize3d_data_graph_matplotlib(input_data_file, target_column_visualize):
    all_columns_list = input_data_file.columns.values.tolist()
    output_column = target_column_visualize
    all_columns_list.remove(output_column)
    outputs_unique = list(set(input_data_file[output_column].astype(str).unique()))
    all_colors = ['#FF1111', '#11FF11', '#1111FF', '#111111', '#FFFF11',
                  '#FF660D', '#CD19FF', '#19AFFF', '#00A0FF', '#E80C1A',
                  '#8F0078', '#008C65', '#5D9100', '#99690B', '#3C8E8F',
                  '#FF7070', '#70FF70', '#7070FF', '#707070', '#FFFE70']
    colors = all_colors[:len(outputs_unique)]
    colors_table = pd.DataFrame({str(output_column): outputs_unique, 'Colors': colors})

    combinations = []
    for l in range(len(all_columns_list)):
        for i in range(1, len(all_columns_list) - l):
            for z in range(1, len(all_columns_list) - i - l):
                combinations.append([l, l + i, l + i + z])

    data_matrix = input_data_file[all_columns_list].values
    output_column_values = input_data_file[output_column].astype(str).values
    data_matrix = np.c_[data_matrix, np.matrix(output_column_values).T]

    fig = plt.figure(num='Visualize Columns Pairs vs Output')

    def calc_cur_combination_coord(i):
        first = combinations[i][0]
        second = combinations[i][1]
        third = combinations[i][2]
        cur_columns = [all_columns_list[first], all_columns_list[second], all_columns_list[third],
                       output_column, 'Label']

        label_column = np.core.defchararray.add(np.asarray(data_matrix[:, first]).reshape(-1).astype(str),
                                                np.asarray(data_matrix[:, second]).reshape(-1).astype(str))

        label_column = np.core.defchararray.add(label_column,
                                                np.asarray(data_matrix[:, third]).reshape(-1).astype(str))

        cur_columns_matrix = data_matrix[:, [first, second, third, len(np.asarray(data_matrix[:1, :]).reshape(-1)) - 1]]
        cur_columns_matrix = np.c_[cur_columns_matrix, np.matrix(label_column).T]
        cur_columns_df = pd.DataFrame(cur_columns_matrix, columns=cur_columns)
        cur_columns_pvt_1 = cur_columns_df.groupby(cur_columns).size().reset_index(name='Outputs_XYZ')
        cur_columns_pvt_2 = cur_columns_df.groupby(cur_columns[-1]).size().reset_index(name='XYZ')
        cur_columns_pvt = pd.merge(cur_columns_pvt_1, cur_columns_pvt_2, on=['Label'])
        cur_columns_pvt = pd.merge(cur_columns_pvt, colors_table, on=[output_column])
        cur_columns_pvt['Scale'] = cur_columns_pvt['Outputs_XYZ'] / cur_columns_pvt['XYZ'].max() * 1000
        label_unique = cur_columns_pvt['Scale'].unique()
        xy_counts = cur_columns_pvt['Outputs_XYZ'].unique()
        label_unique = sorted(label_unique, key=int, reverse=True)
        xy_counts = sorted(xy_counts, key=int, reverse=True)

        cur_label_scale_values_list = []
        cur_label_colors_values_list = []
        cur_label_X_values_list = []
        cur_label_Y_values_list = []
        cur_label_Z_values_list = []
        for i in range(len(label_unique)):
            cur_label_values = cur_columns_pvt.loc[cur_columns_pvt['Scale'] == label_unique[i]]

            cur_label_scale_values_list.append(cur_label_values['Scale'].tolist())
            cur_label_colors_values_list.append(cur_label_values['Colors'].tolist())
            cur_label_X_values_list.append(cur_label_values[all_columns_list[first]].tolist())
            cur_label_Y_values_list.append(cur_label_values[all_columns_list[second]].tolist())
            cur_label_Z_values_list.append(cur_label_values[all_columns_list[third]].tolist())

        return [cur_label_X_values_list, cur_label_Y_values_list, cur_label_Z_values_list,
                cur_label_colors_values_list, cur_label_scale_values_list, xy_counts,
                first, second, third, label_unique]

    cur_scatter_data = calc_cur_combination_coord(0)

    plt.subplots_adjust(left=0.1, right=0.90)

    def new_scatters(cur_scatter_data):
        global fig_legend
        for l in range(len(cur_scatter_data[9])):
            ax.scatter(cur_scatter_data[0][l], cur_scatter_data[1][l],
                       cur_scatter_data[2][l], c=cur_scatter_data[3][l],
                       s=cur_scatter_data[4][l], alpha=0.7, edgecolors=None)

        legend_elements = [Line2D([0], [0], marker='o', color='grey', label=str(cur_scatter_data[5][i]),
                                  markerfacecolor='grey', markersize=cur_scatter_data[9][i] / 1000 * 30) for i in
                           range(len(cur_scatter_data[9]))]
        fig_legend = fig.legend(handles=legend_elements)

        outputs_elements = [Line2D([0], [0], marker='o', color=colors[i], label=outputs_unique[i],
                                   markerfacecolor=colors[i], markersize=10) for i in range(len(outputs_unique))]
        ax.legend(handles=outputs_elements, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=5, mode="expand", borderaxespad=0.)

        ax.set_xlabel(str(all_columns_list[cur_scatter_data[6]]))
        ax.set_ylabel(str(all_columns_list[cur_scatter_data[7]]))
        ax.set_zlabel(str(all_columns_list[cur_scatter_data[8]]))

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    ax = fig.add_subplot(111, projection='3d')
    plt.suptitle('Target Column: ' + str(output_column))
    new_scatters(cur_scatter_data)

    class Index(object):
        ind = 0
        def next(self, event):
            if self.ind <= len(combinations)-1:
                self.ind += 1
            else:
                self.ind = 0
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            fig_legend.remove()
            Axes.clear(ax)
            new_scatters(cur_scatter_data)
            plt.draw()

        def prev(self, event):
            if self.ind > 0:
                self.ind -= 1
            else:
                self.ind = len(combinations)-1
            cur_scatter_data = calc_cur_combination_coord(self.ind)
            fig_legend.remove()
            Axes.clear(ax)
            new_scatters(cur_scatter_data)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.01, 0.05, 0.08, 0.05])
    axnext = plt.axes([0.91, 0.05, 0.08, 0.05])
    bnext = pltButton(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = pltButton(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()


def visualize_data(data, target, type='2D'):
    if type == '2D':
        visualize_data_graph(data, target)
    if type == 'Pseudo-3D':
        visualize3d_data_graph_matplotlib(data, target)
    if type == '3D':
        visualize3d_data_graph(data, target)
