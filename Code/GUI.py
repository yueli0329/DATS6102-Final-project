import sys

#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

from scipy import interp
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler


# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt



#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'


class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the Match Time dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
        self.groupBox1Layout = QGridLayout()  # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        ### list feature here
        self.label1 = QLabel("Blue Gold & RedGold")
        self.label2 = QLabel("Blue Minions Killed & Red Minions Killed")
        self.label3 = QLabel("Blue Jungle Minions Killed & Red Jungle Minions Killed")
        self.label4 = QLabel("Blue AvgLevel & Red AvgLevel")
        self.label5 = QLabel("Blue Herald Kills & Red Herald Kills")
        self.label6 = QLabel("Blue Towers Destroyed & Red Towers Destroyed")
        self.label7 = QLabel("Blue Champ Kills & Red Champ Kills")


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.label1, 0, 0)
        self.groupBox1Layout.addWidget(self.label2, 0, 1)
        self.groupBox1Layout.addWidget(self.label3, 3, 0)
        self.groupBox1Layout.addWidget(self.label4, 1, 1)
        self.groupBox1Layout.addWidget(self.label5, 2, 0)
        self.groupBox1Layout.addWidget(self.label6, 2, 1)
        self.groupBox1Layout.addWidget(self.label7, 1, 0)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 5, 0)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)


        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 1, 1)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.layout.addWidget(self.groupBoxG3, 1, 0 )

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''
        # list feature over there

        vtest_per = float(self.txtPercentTest.text())

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt = X
        y_dt = y

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)

        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # %%----------------------------------------------------------------------------
        ## for test
        X_test = scaler.transform(X_test)

        # perform training with entropy.
        # Decision tree with entropy

        # specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=100)

        # perform training
        self.clf_rf.fit(X_train, y_train)

        # -----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)



        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = ['', 'Blue Win', 'Red Win']

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, X_dt.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance)
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()


class DecisionTree(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()

        self.Title ="Decision Tree Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Decision Tree Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9], self)
        self.feature10 = QCheckBox(features_list[10], self)
        self.feature11 = QCheckBox(features_list[11], self)
        self.feature12 = QCheckBox(features_list[12], self)
        self.feature13 = QCheckBox(features_list[13], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(True)
        self.feature10.setChecked(True)
        self.feature11.setChecked(True)
        self.feature12.setChecked(True)
        self.feature13.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("8")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,0,2)
        self.groupBox1Layout.addWidget(self.feature3,1,0)
        self.groupBox1Layout.addWidget(self.feature4,1,1)
        self.groupBox1Layout.addWidget(self.feature5,1,2)
        self.groupBox1Layout.addWidget(self.feature6,2,0)
        self.groupBox1Layout.addWidget(self.feature7,2,1)
        self.groupBox1Layout.addWidget(self.feature8,2,2)
        self.groupBox1Layout.addWidget(self.feature9,3,0)
        self.groupBox1Layout.addWidget(self.feature10,3,1)
        self.groupBox1Layout.addWidget(self.feature11,3,2)
        self.groupBox1Layout.addWidget(self.feature12,4,0)
        self.groupBox1Layout.addWidget(self.feature13,4,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest,5,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,5,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,6,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,6,1)
        self.groupBox1Layout.addWidget(self.btnExecute,5,2)
        self.groupBox1Layout.addWidget(self.btnDTFigure,6,2)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        #::---------------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------------





        ## End of elements o the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,1,0)
        self.layout.addWidget(self.groupBox2,0,1)



        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):
        '''
        Decision Tree Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = X[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[7]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[7]]],axis=1)


        if self.feature8.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[8]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[8]]],axis=1)



        if self.feature9.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[9]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[9]]],axis=1)


        if self.feature10.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[10]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[10]]],axis=1)


        if self.feature11.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[11]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[11]]],axis=1)



        if self.feature12.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[12]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[12]]],axis=1)



        if self.feature13.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = X[features_list[13]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, X[features_list[13]]],axis=1)




        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100


        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = y


        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)


        class_le = StandardScaler()

        X_train = class_le.fit_transform(X_train)
        X_test = class_le.transform(X_test)


        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=vmax_depth, min_samples_leaf=40)

        # Performing training
        self.clf_entropy.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)
        y_pred_score = self.clf_entropy.predict_proba(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))


        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['','Blue Win', 'Red Win']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_entropy.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

        #::-----------------------------------------------------
        # Graph 2 -- ROC Cure
        #::-----------------------------------------------------




    def view_tree(self):
        '''
        Executes the graphviz to create a tree view of the information
         then it presents the graphic in a pdf formt using webbrowser
        :return:None
        '''
        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.list_corr_features.columns, out_file=None)


        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')

class LogisticRegression(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(LogisticRegression, self).__init__()

        self.Title ="Logistic Regression"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------
        self.btnExecute = QPushButton("Execute LR")
        self.btnExecute.clicked.connect(self.update)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.btnExecute)
        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)




        ## End of elements o the dashboard


        self.layout.addWidget(self.groupBoxG1, 0, 0)
        self.layout.addWidget(self.groupBox2, 0, 1)


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Logist Regression Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        # We assign the values to X and y to run the algorithm

        X_dt = X
        y_dt = y

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.3, random_state=100)

        class_le = StandardScaler()

        X_train = class_le.fit_transform(X_train)
        X_test = class_le.fit_transform(X_test)

        # perform training with entropy.
        # Decision tree with entropy
        self.lr = LR(C=0.5, max_iter=500)

        # Performing training
        self.lr.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_lr = self.lr.predict(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_lr)

        self.ff_class_rep = classification_report(y_test, y_pred_lr)
        self.txtResults.appendPlainText(self.ff_class_rep)


        # accuracy score

        self.df_accuracy_score = accuracy_score(y_test, y_pred_lr) * 100
        self.txtAccuracy.setText(str(self.df_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['', 'Blue Win', 'Red Win']

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.lr.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------



class KNN(QMainWindow):
    #::----------------------
    # Implementation of  KNN Algorithm using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNN, self).__init__()

        self.Title ="KNN"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)





        self.btnExecute = QPushButton("Execute KNN")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.btnExecute)
        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        ## End of elements o the dashboard

        
        self.layout.addWidget(self.groupBoxG1, 0, 0)
        self.layout.addWidget(self.groupBox2, 0, 1)


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        KNN Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''

        

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        

        # We assign the values to X and y to run the algorithm

        X_dt = X
        y_dt = y

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.3, random_state=100)

        class_le = StandardScaler()

        X_train = class_le.fit_transform(X_train)
        X_test = class_le.fit_transform(X_test)

        # perform training with entropy.
        # Decision tree with entropy
        self.knn = KNeighborsClassifier()

        # Performing training
        self.knn.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_knn = self.knn.predict(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_knn)

        # clasification report

        self.df_class_rep = classification_report(y_test, y_pred_knn)
        self.txtResults.appendPlainText(self.df_class_rep)

        # accuracy score

        self.df_accuracy_score = accuracy_score(y_test, y_pred_knn) * 100
        self.txtAccuracy.setText(str(self.df_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['', 'Blue Win', 'Red Win']

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.knn.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------


class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)


class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvaas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 500
        self.top = 200
        self.Title = 'Target Distribution '
        self.width = 700
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=7, height=4)
        self.m.move(0, 30)

class CorrelationPlot(QMainWindow):
    #;:-----------------------------------------------------------------------
    # This class creates a canvas to draw a correlation plot
    # It presents all the features plus the happiness score
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::-----------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        #::--------------------------------------------------------
        super(CorrelationPlot, self).__init__()

        self.Title = 'Correlation Plot'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Creates the canvas and elements of the canvas
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()


        self.groupBox2 = QGroupBox('Correlation Plot')
        self.groupBox2Layout= QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox2Layout.addWidget(self.canvas)

        self.layout.addWidget(self.groupBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(900, 700)
        self.show()
        self.update()

    def update(self):

        #::------------------------------------------------------------
        # Populates the elements in the canvas using the values
        # chosen as parameters for the correlation plot
        #::------------------------------------------------------------
        self.ax1.clear()

        vsticks = ["dummy"]
        vsticks1 = list(X.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = X.corr()
        self.ax1.matshow(res_corr, cmap= plt.cm.get_cmap('Blues', 28))
        self.ax1.set_yticklabels(vsticks1)
        self.ax1.set_xticklabels(vsticks1,rotation = 90)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()



class MatchGraphs(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the features
    #    _init_
    #   update
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a dotplot
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------
        super(MatchGraphs, self).__init__()

        self.Title = "Features vs. Team Win"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(['Team Gold', 'MinionsKilled', 'JungleMinionsKilled',
       'AvgLevel', 'ChampKills', 'HeraldKills','TowersDestroyed'])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")


        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Index for subplots"))
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        self.ax1.clear()

        y_1 = y
        # (['blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled',
        #   'blueAvgLevel', 'redGold', 'redMinionsKilled', 'redJungleMinionsKilled',
        #   'redAvgLevel', 'blueChampKills', 'blueHeraldKills',
        #   'blueTowersDestroyed', 'redChampKills', 'redHeraldKills',
        #   'redTowersDestroyed'])

        cat1 = self.dropdown1.currentText()
        if cat1 =='Team Gold':
            x0_ax = y_1.unique()
            y0_ax = df_f.groupby('blue_win')['redGold'].mean()
            y1_ax = df_f.groupby('blue_win')['blueGold'].mean()
            X_axis = np.arange(len(x0_ax))
            self.ax1.bar(x0_ax - 0.2, y0_ax, 0.4, label='Red Gold')
            self.ax1.bar(x0_ax + 0.2, y1_ax, 0.4, label='Blue Gold')
            vtitle = "Match result vs. "+ cat1+ " in First 15 mins"
            self.ax1.set_title('The bar plot of Team Gold')
            self.ax1.set_xlabel("Red Win                                          Blue Win")
            self.ax1.set_ylabel("Mean of Team Gold")
            self.ax1.legend()
            self.ax1.grid(True)

        if cat1 == 'MinionsKilled':
            x0_ax = y_1.unique()
            y0_ax = df_f.groupby('blue_win')['redMinionsKilled'].mean()
            y1_ax = df_f.groupby('blue_win')['blueMinionsKilled'].mean()
            X_axis = np.arange(len(x0_ax))
            self.ax1.bar(x0_ax - 0.2, y0_ax, 0.4, label='Red Minions Killed')
            self.ax1.bar(x0_ax + 0.2, y1_ax, 0.4, label='Blue Minions Killed')
            #self.ax1.set_xticklabels(X_axis, ['Red win', 'Blue win'])
            vtitle = "Match result vs. " + cat1 + " in First 15 mins"
            self.ax1.set_title('The bar plot of Minions Killed  ')
            self.ax1.set_xlabel("Red Win                                          Blue Win")
            self.ax1.set_ylabel("Mean of Minions Killed")
            self.ax1.legend(loc='upper center')
            self.ax1.grid(True)

        if cat1 == 'JungleMinionsKilled':
            x0_ax = y_1.unique()
            y0_ax = df_f.groupby('blue_win')['redJungleMinionsKilled'].mean()
            y1_ax = df_f.groupby('blue_win')['blueJungleMinionsKilled'].mean()
            X_axis = np.arange(len(x0_ax))
            self.ax1.bar(x0_ax - 0.2, y0_ax, 0.4, label='Red Jungle Minions Killed')
            self.ax1.bar(x0_ax + 0.2, y1_ax, 0.4, label='Blue Jungle Minions Killed')
            # self.ax1.set_xticklabels(X_axis, ['Red win', 'Blue win'])
            vtitle = "Match result vs. " + cat1 + " in First 15 mins"
            self.ax1.set_title('The bar plot of Jungle Minions Killed')
            self.ax1.set_ylabel("Mean of Jungle Minions Killed")
            self.ax1.set_xlabel("Red Win                                          Blue Win")
            self.ax1.legend(loc='upper center')
            self.ax1.grid(True)


        if cat1 == 'AvgLevel':
            x0_ax = y_1.unique()
            y0_ax = df_f.groupby('blue_win')['redAvgLevel'].mean()
            y1_ax = df_f.groupby('blue_win')['blueAvgLevel'].mean()
            X_axis = np.arange(len(x0_ax))
            self.ax1.bar(x0_ax - 0.2, y0_ax, 0.4, label='Red AvgLevel')
            self.ax1.bar(x0_ax + 0.2, y1_ax, 0.4, label='Blue AvgLevel')
            self.ax1.set_xticklabels(X_axis, label='Red win')
            vtitle = "Match result vs. " + cat1 + " in First 15 mins"
            self.ax1.set_title('The bar plot of AvgLevel')
            self.ax1.set_ylabel("Mean of AvgLevel")
            self.ax1.set_xlabel("Red Win                                          Blue Win")
            self.ax1.legend(loc='upper center')
            self.ax1.grid(True)


        if cat1 == 'ChampKills':
            x0_ax = y_1.unique()
            y0_ax = df_f.groupby('blue_win')['redChampKills'].mean()
            y1_ax = df_f.groupby('blue_win')['blueChampKills'].mean()
            X_axis = np.arange(len(x0_ax))
            self.ax1.bar(x0_ax - 0.2, y0_ax, 0.4, label='Red Champ Kills')
            self.ax1.bar(x0_ax + 0.2, y1_ax, 0.4, label='Blue Champ Kills')
            #self.ax1.set_xticklabels('Red win', 'Blue win')
            vtitle = "Match result vs. " + cat1 + " in First 15 mins"
            self.ax1.set_title('The bar plot of Champ Kills')
            self.ax1.set_ylabel("Mean of Champ Kills")
            self.ax1.set_xlabel("Red Win                                          Blue Win")
            self.ax1.legend()
            self.ax1.grid(True)


        if cat1 == 'HeraldKills':
            x0_ax = y_1.unique()
            y0_ax = df_f.groupby('blue_win')['redHeraldKills'].mean()
            y1_ax = df_f.groupby('blue_win')['blueHeraldKills'].mean()
            X_axis = np.arange(len(x0_ax))
            self.ax1.bar(x0_ax - 0.2, y0_ax, 0.4, label='Red Herald Kills')
            self.ax1.bar(x0_ax + 0.2, y1_ax, 0.4, label='Blue Herald Kills')
            #self.ax1.set_xticklabels('Red win', 'Blue win')
            vtitle = "Match result vs. " + cat1 + " in First 15 mins"
            self.ax1.set_title('The bar plot of HeraldKills')
            self.ax1.set_ylabel("Mean of Champ Kills")
            self.ax1.set_xlabel("Red Win                                          Blue Win")
            self.ax1.legend()
            self.ax1.grid(True)


        if cat1 == 'TowersDestroyed':
            x0_ax = y_1.unique()
            y0_ax = df_f.groupby('blue_win')['redTowersDestroyed'].mean()
            y1_ax = df_f.groupby('blue_win')['blueTowersDestroyed'].mean()
            X_axis = np.arange(len(x0_ax))
            self.ax1.bar(x0_ax - 0.2, y0_ax, 0.4, label='Red Towers Destroyed')
            self.ax1.bar(x0_ax + 0.2, y1_ax, 0.4, label='Blue Towers Destroyed')
            #self.ax1.set_xticklabels('Red win', 'Blue win')
            vtitle = "Match result vs. " + cat1 + " in First 15 mins"
            self.ax1.set_title('The bar plot of Towers Destroyed')
            self.ax1.set_ylabel("Mean of Towers Destroyed")
            self.ax1.set_xlabel("Red Win                                          Blue Win")
            self.ax1.legend(loc='upper center')
            self.ax1.grid(True)




        self.fig.tight_layout()
        self.fig.canvas.draw_idle()




class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'League of Legend Game Result Prediction'
        self.width = 500
        self.height = 300
        self.initUI()


    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)


        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        MLModelMenu = mainMenu.addMenu('ML Models')



        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon(), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------


        EDA1Button = QAction(QIcon(),'Target Distribution', self)
        EDA1Button.setStatusTip('Presents the Predictor distribution')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon(),'Features vs. Target', self)
        EDA2Button.setStatusTip('Features Vs Target')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon(),'Correlation Plot', self)
        EDA4Button.setStatusTip('Features Correlation Plot')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)


        #::--------------------------------------------------
        # ML Models for prediction
        # There are four models
        #       Decision Tree
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel1Button = QAction(QIcon(), 'Decision Tree', self)
        MLModel1Button.setStatusTip('ML algorithm with Entropy ')
        MLModel1Button.triggered.connect(self.MLDT)


        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        #::------------------------------------------------------
        # Logistic Regression
        #::------------------------------------------------------
        MLModel3Button = QAction(QIcon(), 'Logistic Regression', self)
        MLModel3Button.setStatusTip('Logistic Regression')
        MLModel3Button.triggered.connect(self.MLLR)
        MLModelMenu.addAction(MLModel3Button)


        #::------------------------------------------------------
        # KNN
        #::------------------------------------------------------
        MLModel4Button = QAction(QIcon(), 'KNN', self)
        MLModel4Button.setStatusTip('KNN')
        MLModel4Button.triggered.connect(self.MLKNN)


        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel3Button)
        MLModelMenu.addAction(MLModel2Button)
        MLModelMenu.addAction(MLModel4Button)

        self.dialogs = list()


    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the blue win
        # X was populated in the method df_f()
        # at the start of the application
        #::------------------------------------------------------
        dialog = CanvasWindow(self)
        dialog.m.plot()
        dialog.m.ax.hist(y, facecolor='green')
        dialog.m.ax.set_title('Team Win Distribution')
        dialog.m.ax.set_xlabel("Red Win                                                                                 Blue Win")
        dialog.m.ax.set_ylabel("The number of the Matching Game")
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        # This class creates a graph using the features in the dataset
        # blue win vs. other feature
        #::---------------------------------------------------------
        dialog = MatchGraphs()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()


    def MLDT(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the   dataset
        #::-------------------------------------------------------------
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the   dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def MLLR(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the   dataset
        #::-------------------------------------------------------------
        dialog = LogisticRegression()
        self.dialogs.append(dialog)
        dialog.show()

    def MLKNN(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the   dataset
        #::-------------------------------------------------------------
        dialog = KNN()
        self.dialogs.append(dialog)
        dialog.show()


def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def MatchTimelinesFirst15():
    #::--------------------------------------------------
    # Loads the dataset MatchTimelinesFirst15.csv
    # Loads the dataset MatchTimelinesFirst15
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------
    global df_f
    global X
    global y
    global features_list
    global class_names
    df_f = pd.read_csv('data_team06.csv')
    X= df_f.drop(labels="blue_win", axis=1)
    y= df_f['blue_win']
    features_list = ['blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled',
       'blueAvgLevel', 'redGold', 'redMinionsKilled', 'redJungleMinionsKilled',
       'redAvgLevel', 'blueChampKills', 'blueHeraldKills',
       'blueTowersDestroyed', 'redChampKills', 'redHeraldKills',
       'redTowersDestroyed']
    class_names = ['Blue Win', 'Red Win']


if __name__ == '__main__':
    MatchTimelinesFirst15()
    main()