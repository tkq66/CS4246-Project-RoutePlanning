import sys
from enum import Enum

import numpy as np
import scipy.sparse as sparse

import mdptoolbox.mdp as mdp

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
        
grid_width = 800
grid_height = 800
window_width = 1000
window_height = 800
block_size = 0

col_count = 5
row_count = 5
grid_map_size = col_count * row_count

default_reward = -0.1

class Actions(Enum):
    NORTH = 0
    WEST = 1
    EAST = 2
    SOUTH = 3
allowed_actions_count = 4

driver = None
client = None
dest = None
R = None
P = None

def initData():
    global driver
    global client
    global dest
    global R
    global P
    driver = {
        "col": 0,
        "row": 0,
        "index": 0
    }
    client = {
        "col": 0,
        "row": 0,
        "index": 0
    }
    dest = {
        "col": 0,
        "row": 0,
        "index": 0
    }
    R = np.full((grid_map_size, allowed_actions_count), default_reward)
    P = np.zeros((allowed_actions_count, grid_map_size, grid_map_size))
    P[Actions.NORTH.value] = np.vstack(
                                (np.eye(N=col_count,
                                        M=grid_map_size),
                                 np.eye(N=(grid_map_size-col_count),
                                        M=grid_map_size)))
    P[Actions.EAST.value] = sparse.block_diag(
                                [np.pad(
                                    np.eye(N=(col_count-1),
                                           M=col_count,
                                           k=1),
                                    ((0, 1), (0, 0)),
                                    'edge')
                                ]*row_count).toarray()
    P[Actions.WEST.value] = np.flip(P[Actions.EAST.value], (0, 1))
    P[Actions.SOUTH.value] = np.flip(P[Actions.NORTH.value], (0, 1))
    P *= 0.9
    stationary_transition = np.repeat(np.identity(grid_map_size)[:, :, np.newaxis],
                                      allowed_actions_count,
                                      axis=2).T * 0.1
    P += stationary_transition
    
def update_event(pos_index, severity):
    global P
    active_index = np.array(
                        np.nonzero(
                            P[:,
                              :,
                              pos_index]))
    ext_active_index = active_index[:, 
                                    active_index[1] != pos_index]
    int_active_index = active_index[:, 
                                    np.logical_and(
                                        active_index[1] == pos_index, 
                                        P[active_index[0], 
                                          active_index[1], 
                                          active_index[1]] != 1)]
    int_row_nonzero_index = np.array(
                                np.nonzero(
                                    P[int_active_index[0], 
                                      int_active_index[1], 
                                      :]))
    int_secondary_index = int_row_nonzero_index[1, int_row_nonzero_index[1] != pos_index]
    P[ext_active_index[0], 
      ext_active_index[1], 
      pos_index] = 1 - severity
    P[ext_active_index[0], 
      ext_active_index[1], 
      ext_active_index[1]] = severity
    P[int_active_index[0], 
      int_active_index[1], 
      pos_index] = severity
    P[int_active_index[0], 
      int_active_index[1], 
      int_secondary_index] = 1 - severity

def update_reward(pos_index, reward):
    global R
    R[pos_index, :] = reward
    
class SimulationSetting(QWidget):
    simulationRan = pyqtSignal(dict)
    showReward = pyqtSignal()
    showTransition = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.steps = 0
        self.isPickedUp = False
        self.isArrivedDest = False
        self.isShowReward = True
        self.isShowTransition = False
        self.actionType = Actions.NORTH

        self.runBtn = None
        self.stepsCounter = None
        self.pickedUpStatus = None
        self.arrivedDestStatus = None
        self.showRewardRadio = None
        self.showTransitionRadio = None
        self.actionTransitionTypeCombobox = None
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout()
        group = QGroupBox("Simulation")
        fl = QFormLayout()
        
        self.runBtn = QPushButton("Run")
        self.runBtn.clicked.connect(self.run_simulation)
        
        self.stepsCounter = QLabel(str(self.steps))
        
        self.pickedUpStatus = QLabel(str(self.isPickedUp))
        self.pickedUpStatus.setStyleSheet("color: red")
        
        self.arrivedDestStatus = QLabel(str(self.isArrivedDest))
        self.arrivedDestStatus.setStyleSheet("color: red")
        
        self.showRewardRadio = QRadioButton("Show Reward")
        self.showRewardRadio.setChecked(self.isShowReward)
        self.showRewardRadio.toggled.connect(self.show_reward)
        
        self.showTransitionRadio = QRadioButton("Show Transition")
        self.showTransitionRadio.setChecked(self.isShowTransition)
        self.showTransitionRadio.toggled.connect(self.show_transition)
        
        self.actionTransitionTypeCombobox = QComboBox()
        self.actionTransitionTypeCombobox.addItems([Actions.NORTH.name, Actions.WEST.name, Actions.EAST.name, Actions.SOUTH.name])
        self.actionTransitionTypeCombobox.currentIndexChanged.connect(self.show_transition)
        
        fl.addRow(self.runBtn)
        fl.addRow(QLabel("Steps:"), self.stepsCounter)
        fl.addRow(QLabel("Picked Up Client?"), self.pickedUpStatus)
        fl.addRow(QLabel("Arrived Destination?"), self.arrivedDestStatus)
        fl.addRow(self.showRewardRadio)
        fl.addRow(self.showTransitionRadio)
        fl.addRow(self.actionTransitionTypeCombobox)
        group.setLayout(fl)
        layout.addWidget(group)
        self.setLayout(layout)
    
    def run_simulation(self, *args, **kwargs):
        mdp_planner = mdp.PolicyIteration(P, R, 0.9)
        mdp_planner.run()
        mdp_policy = mdp_planner.policy
        driver_policy = mdp_policy[driver["index"]]
        action = Actions(driver_policy)
        ideal_dest = {
            "col": driver["col"] + 1 if action is Actions.EAST and driver["col"] != (col_count - 1) else driver["col"] - 1 if action is Actions.WEST and driver["col"] != 0 else driver["col"],
            "row": driver["row"] + 1 if action is Actions.SOUTH and driver["row"] != (row_count - 1) else driver["row"] - 1 if action is Actions.NORTH and driver["row"] != 0 else driver["row"]
        }
        ideal_dest_index = (row_count * ideal_dest["row"]) + ideal_dest["col"]
        action_prob = P[action.value, driver["index"], ideal_dest_index]
        policy_succeed = np.random.choice([0,1], 1, p=[1 - action_prob, action_prob])[0]
        self.simulation_detail = {
            "action": action,
            "source": {
                "col": driver["col"],
                "row": driver["row"],
                "index": driver["index"]
            },
            "dest": {
                "col": ideal_dest["col"] if policy_succeed else driver["col"],
                "row":ideal_dest["row"] if policy_succeed else driver["row"],
                "index": ideal_dest_index if policy_succeed else driver["index"]
            }
        }
        
        self.steps = self.steps + 1
        self.stepsCounter.setText(str(self.steps))
        
        self.isPickedUp = True if self.isPickedUp or self.simulation_detail["dest"]["index"] == client["index"] else False
        self.pickedUpStatus.setText(str(self.isPickedUp))
        self.pickedUpStatus.setStyleSheet("color: {}".format("green" if self.isPickedUp else "red"))
        
        print("dest", dest)
        print("sim", self.simulation_detail["dest"])
        self.isArrivedDest = True if self.isArrivedDest or self.simulation_detail["dest"]["index"] == dest["index"] else False
        self.arrivedDestStatus.setText(str(self.isArrivedDest))
        self.arrivedDestStatus.setStyleSheet("color: {}".format("green" if self.isArrivedDest else "red"))
        
        self.simulationRan.emit(self.simulation_detail)
    
    def show_reward(self, *args, **kwargs):
        if self.showTransitionRadio.isChecked():
            return
        self.showReward.emit()
        
    def show_transition(self, *args, **kwargs):
        if self.showRewardRadio.isChecked():
            return
        action = Actions[self.actionTransitionTypeCombobox.currentText()]
        action_index = action.value
        transition_array = P[action_index, driver["index"]]
        self.transition_details = {
            "probability_array": transition_array,
            "action": action,
            "source": driver
        }
        self.showTransition.emit(self.transition_details)
        
    def reset(self):
        self.steps = 0
        self.isPickedUp = False
        self.isArrivedDest = False
        self.isShowReward = True
        self.isShowTransition = False
        
        self.stepsCounter.setText(str(self.steps))
        self.pickedUpStatus.setText(str(self.isPickedUp))
        self.pickedUpStatus.setStyleSheet("color: red")
        self.arrivedDestStatus.setText(str(self.isArrivedDest))
        self.arrivedDestStatus.setStyleSheet("color: red")
        self.showRewardRadio.setChecked(self.isShowReward)
        self.showTransitionRadio.setChecked(self.isShowTransition)
        self.actionTransitionTypeCombobox.setCurrentText(Actions.NORTH.name)

class UnitPriceSetting(QWidget):
    unitPriceChanged = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.unitPriceDetail = None
        self.unitPrice = None
        self.addBtn = None
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout()
        group = QGroupBox("Unit Price")
        fl = QFormLayout()
        
        self.unitPrice = QLineEdit()
        self.unitPrice.setValidator(QDoubleValidator(0.00, 1000.00, 1))
        self.unitPrice.setPlaceholderText("Number {} - {}".format(0.0, 1000.0))
        
        self.addBtn = QPushButton("Add")
        self.addBtn.clicked.connect(self.add_unit_price)
        
        fl.addRow(QLabel("Unit Price"), self.unitPrice)
        fl.addRow(self.addBtn)
        group.setLayout(fl)
        layout.addWidget(group)
        self.setLayout(layout)
    
    def add_unit_price(self, *args, **kwargs):
        self.unitPriceDetail = float(self.unitPrice.text())
        self.unitPriceChanged.emit(self.unitPriceDetail)
    
    def reset(self):
        self.unitPrice.clear()

class PosSetting(QWidget):
    posChanged = pyqtSignal(dict)
    
    def __init__(self, title, onChangeCallback=None):
        super().__init__()
        self.title = title
        self.onChangeCallback = onChangeCallback
        self.pos = {
            "col": 0,
            "row": 0
        }
        self.col = None
        self.row = None
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout()
        group = QGroupBox(self.title)
        fl = QFormLayout()
        
        self.col = QLineEdit()
        self.col.setValidator(QIntValidator(0, col_count - 1))
        self.col.setPlaceholderText("Integer {} - {}".format(0, col_count - 1))
        self.col.textChanged.connect(self.col_val_changed)
        
        self.row = QLineEdit()
        self.row.setValidator(QIntValidator(0, row_count - 1))
        self.row.setPlaceholderText("Integer {} - {}".format(0, row_count - 1))
        self.row.textChanged.connect(self.row_val_changed)
        
        fl.addRow(QLabel("Column"), self.col)
        fl.addRow(QLabel("Row"), self.row)
        group.setLayout(fl)
        layout.addWidget(group)
        self.setLayout(layout)
    
    def col_val_changed(self, *args, **kwargs):
        self.pos["col"] = int(0 if not self.sender().text() else self.sender().text())
        if self.onChangeCallback:
            self.onChangeCallback(self.pos)
        self.posChanged.emit(self.pos)
    
    def row_val_changed(self, *args, **kwargs):
        self.pos["row"] = int(0 if not self.sender().text() else self.sender().text())
        if self.onChangeCallback:
            self.onChangeCallback(self.pos)
        self.posChanged.emit(self.pos)
    
    def reset(self):
        self.col.clear()
        self.row.clear()
        
class IncidentSetting(QWidget):
    incidentAdded = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.incidentDetails = []
        self.incidentDetail = {
            "name": None,
            "severity": None,
            "col": None,
            "row": None,
        }
        self.incidentName = None
        self.severity = None
        self.col = None
        self.row = None
        self.addBtn = None
        self.tableRecord = None
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout()
        group = QGroupBox("Incidents")
        fl = QFormLayout()
        self.incidentName = QLineEdit()
        
        self.severity = QLineEdit()
        self.severity.setValidator(QDoubleValidator(0.00, 1.00, 1))
        self.severity.setPlaceholderText("Number {} - {}".format(0.0, 1.0))
        
        self.col = QLineEdit()
        self.col.setValidator(QIntValidator(0, col_count - 1))
        self.col.setPlaceholderText("Integer {} - {}".format(0, col_count - 1))
        
        self.row = QLineEdit()
        self.row.setValidator(QIntValidator(0, row_count - 1))
        self.row.setPlaceholderText("Integer {} - {}".format(0, row_count - 1))
        
        self.addBtn = QPushButton("Add")
        self.addBtn.clicked.connect(self.addIncident)
        
        self.tableRecord = QTableWidget()
        self.tableRecord.setColumnCount(4)
        self.tableRecord.setHorizontalHeaderLabels(["Name", "Severity", "Col", "Row"]) 

        fl.addRow(QLabel("Incident Name"), self.incidentName)
        fl.addRow(QLabel("Severity"), self.severity)
        fl.addRow(QLabel("Column"), self.col)
        fl.addRow(QLabel("Row"), self.row)
        fl.addRow(self.addBtn)
        fl.addRow(self.tableRecord)
        group.setLayout(fl)
        layout.addWidget(group)
        self.setLayout(layout)
        
    def reset(self):
        self.incidentName.clear()
        self.severity.clear()
        self.col.clear()
        self.row.clear()
        self.tableRecord.setRowCount(0)
    
    def addIncident(self, *args, **kwargs):
        self.incidentDetail = {
            "name": self.incidentName.text(),
            "severity": float(self.severity.text()),
            "col": int(self.col.text()),
            "row": int(self.row.text())
        }
        index = (row_count * self.incidentDetail["row"]) + self.incidentDetail["col"]
        self.incidentDetails.append(self.incidentDetail)
        newRowIndex = self.tableRecord.rowCount()
        self.tableRecord.insertRow(newRowIndex)
        self.tableRecord.setItem(newRowIndex, 0, QTableWidgetItem(str(self.incidentDetail["name"])))
        self.tableRecord.setItem(newRowIndex, 1, QTableWidgetItem(str(self.incidentDetail["severity"])))
        self.tableRecord.setItem(newRowIndex, 2, QTableWidgetItem(str(self.incidentDetail["col"])))
        self.tableRecord.setItem(newRowIndex, 3, QTableWidgetItem(str(self.incidentDetail["row"])))
        update_event(index, self.incidentDetail["severity"])
        self.incidentAdded.emit(self.incidentDetail)
        
class RewardSetting(QWidget):
    rewardAdded = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.rewardDetail = {
            "reward": None,
            "col": None,
            "row": None
        }
        self.reward = None
        self.col = None
        self.row = None
        self.addBtn = None
        self.tableRecord = None
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout()
        group = QGroupBox("Rewards")
        fl = QFormLayout()
        
        self.reward = QLineEdit()
        self.reward.setValidator(QDoubleValidator(0.00, 1000.00, 1))
        self.reward.setPlaceholderText("Number {} - {}".format(0.0, 1000.0))
        
        self.col = QLineEdit()
        self.col.setValidator(QIntValidator(0, col_count - 1))
        self.col.setPlaceholderText("Integer {} - {}".format(0, col_count - 1))
        
        self.row = QLineEdit()
        self.row.setValidator(QIntValidator(0, row_count - 1))
        self.row.setPlaceholderText("Integer {} - {}".format(0, row_count - 1))
        
        self.addBtn = QPushButton("Add")
        self.addBtn.clicked.connect(self.addReward)
        
        self.tableRecord = QTableWidget(grid_map_size, 3)
        self.tableRecord.setHorizontalHeaderLabels(["Reward", "Col", "Row"]) 
        positions = [(i,j) for i in range(row_count) for j in range(col_count)]
        for position in positions:
            array_index = (row_count * position[0]) + position[1]
            self.tableRecord.setItem(array_index, 0, QTableWidgetItem(str(R[array_index, 0])))
            self.tableRecord.setItem(array_index, 1, QTableWidgetItem(str(position[1])))
            self.tableRecord.setItem(array_index, 2, QTableWidgetItem(str(position[0])))

        fl.addRow(QLabel("Reward"), self.reward)
        fl.addRow(QLabel("Column"), self.col)
        fl.addRow(QLabel("Row"), self.row)
        fl.addRow(self.addBtn)
        fl.addRow(self.tableRecord)
        group.setLayout(fl)
        layout.addWidget(group)
        self.setLayout(layout)
        
    def reset(self):
        self.reward.clear()
        self.col.clear()
        self.row.clear()
        for i in range(self.tableRecord.rowCount()):
            self.tableRecord.setItem(i, 0, QTableWidgetItem(str(R[i, 0])))
    
    def addReward(self, *args, **kwargs):
        self.rewardDetail = {
            "reward": float(self.reward.text()),
            "col": int(self.col.text()),
            "row": int(self.row.text())
        }
        item_index = (row_count * self.rewardDetail["row"]) + self.rewardDetail["col"]
        update_reward(item_index, self.rewardDetail["reward"])
        self.tableRecord.setItem(item_index, 0, QTableWidgetItem(str(self.rewardDetail["reward"])))
        self.rewardAdded.emit(self.rewardDetail)
        
class Settings(QWidget):
    reset = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.resetBtn = QPushButton("Reset")
        self.simulation = SimulationSetting()
        self.unitPrice = UnitPriceSetting()
        self.driverPos = PosSetting("Driver Position", self.driverPosChanged)
        self.clientPos = PosSetting("Client Position", self.clientPosChanged)
        self.destPos = PosSetting("Destination Position", self.destPosChanged)
        self.incidents = IncidentSetting()
        self.rewards = RewardSetting()
        self.initUI()
        
    def driverPosChanged(self, pos):
        global driver
        driver = {
            "col": pos["col"],
            "row": pos["row"],
            "index": (row_count * pos["row"]) + pos["col"]
        }
        
    def clientPosChanged(self, pos):
        global client
        client = {
            "col": pos["col"],
            "row": pos["row"],
            "index": (row_count * pos["row"]) + pos["col"]
        }
        
    def destPosChanged(self, pos):
        global dest
        dest = {
            "col": pos["col"],
            "row": pos["row"],
            "index": (row_count * pos["row"]) + pos["col"]
        }
        
    def initUI(self):
        vb_top = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        vb = QVBoxLayout()
        vb.addWidget(self.resetBtn)
        vb.addWidget(self.simulation)
        vb.addWidget(self.unitPrice)
        vb.addWidget(self.driverPos)
        vb.addWidget(self.clientPos)
        vb.addWidget(self.destPos)
        vb.addWidget(self.incidents)
        vb.addWidget(self.rewards)
        vb.setAlignment(Qt.AlignTop)
        settingsContent = QWidget()
        settingsContent.setLayout(vb)

        scroll.setWidget(settingsContent)
        vb_top.addWidget(scroll)
        self.setLayout(vb_top)
        
        self.resetBtn.clicked.connect(self.resetSettings)
    
    @pyqtSlot()
    def resetSettings(self, *args, **kwargs):
        initData()
        self.simulation.reset()
        self.unitPrice.reset()
        self.driverPos.reset()
        self.clientPos.reset()
        self.destPos.reset()
        self.incidents.reset()
        self.rewards.reset()
        self.reset.emit()
        
class Pos(QWidget):
    def __init__(self, x, y, width, height, *args, **kwargs):
        super(Pos, self).__init__(*args, **kwargs)
        self.setFixedSize(QSize(width, height))
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.currentIndex = (row_count * self.y) + self.x
        self.isShowReward = True
        self.isShowTransition = False
        self.reward = default_reward
        self.transition_prob = 0
        self.isDriver = False
        self.isClient = False
        self.isDest = False
        self.isIncident = False
        self.isPast = False

    def paintEvent(self, event):
        p = QPainter()
        p.begin(self)
        r = event.rect()
        p.fillRect(r, QBrush(Qt.yellow if self.isDriver 
                                       else Qt.blue if self.isClient 
                                       else Qt.green if self.isDest
                                       else Qt.red if self.isIncident
                                       else Qt.black if  self.isPast
                                       else Qt.lightGray ))
        p.setPen(Qt.black if not self.isPast else Qt.white)
        p.setFont(QFont("Arial", self.width * 0.3, 0, False))
        p.drawText(r, Qt.AlignCenter, str(self.reward) if self.isShowReward else str(self.transition_prob))  
        p.end()
    
    @pyqtSlot(dict)
    def driverPosChanged(self, *args, **kwargs):
        driverPos = self.sender().pos
        self.isDriver = True if driverPos["row"] == self.y and driverPos["col"] == self.x else False
        self.isClient = False if self.isDriver else self.isClient
        self.isDest = False if self.isDriver else self.isDest
        self.isIncident = False if self.isDriver else self.isIncident
        self.isPast = False if self.isDriver else self.isPast
        self.update()
        
    @pyqtSlot(dict)
    def clientPosChanged(self, *args, **kwargs):
        clientPos = self.sender().pos
        self.isClient = True if clientPos["row"] == self.y and clientPos["col"] == self.x else False
        self.isDriver = False if self.isClient else self.isDriver
        self.isDest = False if self.isClient else self.isDest
        self.isIncident = False if self.isClient else self.isIncident
        self.isPast = False if self.isClient else self.isPast
        self.update()
        
    @pyqtSlot(dict)
    def destPosChanged(self, *args, **kwargs):
        destPos = self.sender().pos
        self.isDest = True if destPos["row"] == self.y and destPos["col"] == self.x else False
        self.isDriver = False if self.isDest else self.isDriver
        self.isClient = False if self.isDest else self.isClient
        self.isIncident = False if self.isDest else self.isIncident
        self.isPast = False if self.isDest else self.isPast
        self.update()
        
    @pyqtSlot(dict)
    def addIncident(self, *args, **kwargs):
        incidentDetail = self.sender().incidentDetail
        self.isIncident = True if self.isIncident or (incidentDetail["row"] == self.y and incidentDetail["col"] == self.x) else False
        self.isDriver = False if self.isIncident else self.isDriver
        self.isClient = False if self.isIncident else self.isClient
        self.isDest = False if self.isIncident else self.isDest
        self.isPast = False if self.isIncident else self.isPast
        self.update()
        
    @pyqtSlot(dict)
    def updateSimulationResult(self, *args, **kwargs):
        global driver
        simulationDetail = self.sender().simulation_detail
        self.isPast = True if self.isPast or (simulationDetail["source"]["row"] == self.y and simulationDetail["source"]["col"] == self.x) else False
        self.isDriver = True if simulationDetail["dest"]["row"] == self.y and simulationDetail["dest"]["col"] == self.x else False
        self.isClient = False if self.isPast or self.isDriver else self.isClient
        self.isDest = False if self.isPast or self.isDriver else self.isDest
        self.isIncident = False if self.isPast or self.isDriver else self.isIncident
        driver = {
            "col": simulationDetail["dest"]["col"],
            "row": simulationDetail["dest"]["row"],
            "index": simulationDetail["dest"]["index"]
        }
        self.update()
        
    @pyqtSlot(dict)
    def addReward(self, *args, **kwargs):
        rewardDetail = self.sender().rewardDetail
        if not (rewardDetail["row"] == self.y and rewardDetail["col"] == self.x):
            return
        self.reward = rewardDetail["reward"]
        self.update()
    
    @pyqtSlot()
    def showReward(self, *args, **kwargs):
        self.isShowReward = True
        self.isShowTransition = False
        self.update()

    @pyqtSlot(dict)
    def showTransition(self, *args, **kwargs):
        transitionDetail = self.sender().transition_details
        self.transition_prob = transitionDetail["probability_array"][self.currentIndex]
        self.isShowReward = False
        self.isShowTransition = True
        self.update()
    
    @pyqtSlot()
    def reset(self, *args, **kwargs):
        self.isShowReward = True
        self.isShowTransition = False
        self.isDriver = False
        self.isClient = False
        self.isDest = False
        self.isIncident = False
        self.isPast = False
        self.update()
        
class Grid(QWidget):
    
    def __init__(self, settings_widget):
        super().__init__()
        self.settings_widget = settings_widget
        self.initUI()
        
    def initUI(self):
        vb = QVBoxLayout()
        grid = QGridLayout()
        self.setLayout(grid)
        
        if (row_count >= col_count):
            block_size = (grid_height / row_count) - 4
        else:
            block_size = (grid_width / col_count) - 4
 
        positions = [(i,j) for i in range(row_count) for j in range(col_count)]
        for position in positions:
            w = Pos(position[1], position[0], block_size, block_size)
            self.settings_widget.simulation.simulationRan.connect(w.updateSimulationResult)
            self.settings_widget.simulation.showReward.connect(w.showReward)
            self.settings_widget.simulation.showTransition.connect(w.showTransition)
            self.settings_widget.driverPos.posChanged.connect(w.driverPosChanged)
            self.settings_widget.driverPos.posChanged.connect(self.settings_widget.simulation.show_transition)
            self.settings_widget.clientPos.posChanged.connect(w.clientPosChanged)
            self.settings_widget.destPos.posChanged.connect(w.destPosChanged)
            self.settings_widget.incidents.incidentAdded.connect(w.addIncident)
            self.settings_widget.incidents.incidentAdded.connect(self.settings_widget.simulation.show_transition)
            self.settings_widget.rewards.rewardAdded.connect(w.addReward)
            self.settings_widget.reset.connect(w.reset)
            grid.addWidget(w, position[0], position[1])
            
            
class Main(QWidget):
    def __init__(self, *args, **kwargs):
        super(Main, self).__init__(*args, **kwargs)
        self.settings = Settings()
        self.grid = Grid(self.settings)
        self.initUI()
        
    def initUI(self):
        hb = QHBoxLayout()
        hb.addWidget(self.grid)
        hb.addWidget(self.settings)
        self.setLayout(hb)
        self.setWindowTitle("MDP Taxi")
        self.center()
        self.show()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
initData()
app = QApplication(sys.argv)
ex = Main()
sys.exit(app.exec_())