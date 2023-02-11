import os
import argparse
import copy

MemSize = 1000  # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.


class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name

        with open(ioDir + "\\imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

        for i in range(len(self.IMem), 1000):
            self.IMem.append('00000000')

    def readInstr(self, ReadAddress):
        # read instruction memory
        # return 32 bit hex val
        instruction = 0
        for address in range(ReadAddress, ReadAddress + 4):
            instruction = instruction * (2 ** 8) + int(self.IMem[address], 2)
        return hex(instruction)[2:]


class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir

        with open(ioDir + "\\dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
        for i in range(len(self.DMem), 1000):
            self.DMem.append('00000000')

    def readData(self, ReadAddress):
        # read data memory
        # return 32 bit hex val
        data = 0
        for address in range(ReadAddress, ReadAddress + 4):
            data = data * (2 ** 8) + int(self.DMem[address], 2)
        data = hex(data)[2:]
        while len(data) < 8:
            data = '0' + data
        return data

    def writeDataMem(self, Address, WriteData):
        # write data into byte addressable memory
        if WriteData >= 0:
            data = bin(WriteData)[2:]
            dataList = ['0' for _ in range(32)]
            t1 = len(data) - 1
            t2 = len(dataList) - 1
            while t1 >= 0:
                dataList[t2] = data[t1]
                t2 = t2 - 1
                t1 = t1 - 1
        else:
            complCode = 2 ** 31 - abs(WriteData)
            data = bin(complCode)[2:]
            dataList = ['0' for _ in range(32)]
            dataList[0] = '1'
            t1 = len(data) - 1
            t2 = len(dataList) - 1
            while t1 >= 0:
                dataList[t2] = data[t1]
                t2 = t2 - 1
                t1 = t1 - 1

        WriteData = ''.join(dataList)
        start = 0
        for sub_address in range(Address, Address + 4):
            self.DMem[sub_address] = WriteData[start:start + 8]
            start = start + 8

    def outputDataMem(self):
        resPath = self.ioDir + "\\" + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])


class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [0x0 for i in range(32)]

    def readRF(self, Reg_addr):
        # Fill in
        return self.Registers[Reg_addr]

    def writeRF(self, Reg_addr, Wrt_reg_data):

        if Reg_addr == 0:
            return 0
        # Fill in
        self.Registers[Reg_addr] = Wrt_reg_data

    def outputRF(self, cycle):
        op = ["-" * 70 + "\n", "State of RF after executing cycle:" + str(cycle) + "\n"]

        for val in self.Registers:
            if val >= 0:
                data = bin(val)[2:]
                dataList = ['0' for _ in range(32)]
                t1 = len(data) - 1
                t2 = len(dataList) - 1
                while t1 >= 0:
                    dataList[t2] = data[t1]
                    t2 = t2 - 1
                    t1 = t1 - 1
            else:
                complCode = 2 ** 31 - abs(val)
                data = bin(complCode)[2:]
                dataList = ['0' for _ in range(32)]
                dataList[0] = '1'
                t1 = len(data) - 1
                t2 = len(dataList) - 1
                while t1 >= 0:
                    dataList[t2] = data[t1]
                    t2 = t2 - 1
                    t1 = t1 - 1

            val = ''.join(dataList)
            op.extend([val + "\n"])

        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)


class State(object):
    def __init__(self):
        self.IF = {"PC": 0, "nop": 0}
        self.ID = {"Instr": 0, "nop": 1}
        self.EX = {"RS1": 0, "RS2": 0, "RS1_val": 0,
                   "RS2_val": 0, "imm": 0, "Operand2FromRS2": 0,  # Operand2FromRS2 = MUX1
                   "DestReg": 0, "AluOperation": 0, "is_branch": 0,  # is_branch = MUX2. If JAL, MUX2 already done in ID
                   "WBEnable": 0, "RdDMem": 0, "WrDMem": 0, "Halt": 0, "nop": 1}
        self.MEM = {"ALUresult": 0, "Store_data": 0, "Wrt_reg_addr": 0, "rd_mem": 0,  # Store_data from RS2_val
                    "wrt_mem": 0, "wrt_enable": 0, "nop": 1}  # wrt_enable is to write back to register
        self.WB = {"Wrt_data": 0, "Wrt_reg_addr": 0, "wrt_enable": 0, "nop": 1}


class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.instrNumber = 0
        self.instruLocationList = []
        self.halt = False
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem


class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + "\\SS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_SS.txt"

    # n bit binary string imm to 10 number
    def binStr2num(self, strData, n):
        if strData[0] == '0':
            data = int(strData[1:], 2)
        else:
            data = int(strData[1:], 2)
            data = 2 ** (n - 1) - data
            data = -1 * data
        return data

    # 32 bit hex string to 10 number
    def hexStr2num(self, strData):
        negativeTrans = {
            '8': '0',
            '9': '1',
            'a': '2',
            'b': '3',
            'c': '4',
            'd': '5',
            'e': '6',
            'f': '7'
        }
        if strData[0] in ['0', '1', '2', '3', '4', '5', '6', '7']:
            data = int(strData, 16)
        else:
            data = negativeTrans[strData[0]] + strData[1:]
            data = int(data, 16)
            data = 2 ** 31 - data
            data = -1 * data
        return data

    # decode the instruction
    def deCodeInstr(self, instruction):
        opCode = instruction[25:]
        func3 = instruction[17:20]
        func7 = instruction[0:7]
        self.state.EX["DestReg"] = int(instruction[20:25], 2)
        self.state.EX["RS1"] = int(instruction[12:17], 2)
        self.state.EX["RS2"] = int(instruction[7:12], 2)

        return opCode, func3, func7

    # get the information
    def control(self, opCode, func3, func7, instruction):
        MemtoReg = 0
        regWrite = 0
        imm = 0

        # ----- R type instruction -----
        if opCode == '0110011':
            self.state.EX['is_branch'] = 0
            self.state.EX['Operand2FromRS2'] = 1
            self.state.EX['RdDMem'] = 0
            self.state.EX['WrDMem'] = 0
            self.state.EX['RS1_val'] = self.myRF.readRF(self.state.EX["RS1"])
            self.state.EX['RS2_val'] = self.myRF.readRF(self.state.EX["RS2"])
            MemtoReg = 0
            regWrite = 1
            imm = 0

            if func3 == '000':
                if func7 == '0000000':  # ADD
                    self.state.EX['AluOperation'] = 0
                else:  # SUB
                    self.state.EX['AluOperation'] = 1
            if func3 == '100':  # XOR
                self.state.EX['AluOperation'] = 2
            if func3 == '110':  # OR
                self.state.EX['AluOperation'] = 3
            if func3 == '111':  # AND
                self.state.EX['AluOperation'] = 4

        # ----- I type instruction ----- (normal)
        if opCode == '0010011':
            self.state.EX['is_branch'] = 0
            self.state.EX['Operand2FromRS2'] = 0
            self.state.EX['RdDMem'] = 0
            self.state.EX['WrDMem'] = 0
            MemtoReg = 0
            regWrite = 1
            imm = self.binStr2num(instruction[0:12], 12)
            self.state.EX['RS1_val'] = self.myRF.readRF(self.state.EX["RS1"])

            if func3 == '000':
                self.state.EX['AluOperation'] = 0
            #    print('addi x%d x%d imm' %(rd,rs1))
            #    print(rs1_val + imm,rs1_val , imm)
            if func3 == '100':
                self.state.EX['AluOperation'] = 2
                # print('xori x%d x%d imm' %(rd,rs1))
                # print(rs1_val ^ imm,rs1_val , imm)
            if func3 == '110':
                self.state.EX['AluOperation'] = 3
                # print('ori x%d x%d imm' %(rd,rs1))
                # print(rs1_val | imm,rs1_val , imm)
            if func3 == '111':
                self.state.EX['AluOperation'] = 4
                # print('andi x%d x%d imm' %(rd,rs1))
                # print(rs1_val & imm,rs1_val , imm)

        # ----- J type instruction -----
        if opCode == '1101111':
            self.state.EX['is_branch'] = 1
            self.state.EX['Operand2FromRS2'] = 0
            self.state.EX['RdDMem'] = 0
            self.state.EX['WrDMem'] = 0
            MemtoReg = 0
            regWrite = 1
            s = instruction[0] + instruction[12:20] + instruction[11] + instruction[1:11] + '0'
            imm = self.binStr2num(s, 21)
            self.state.EX['AluOperation'] = 5
            # print('jal x%d imm' %(rd))
            # print(self.nextState.IF['PC'])

        # ----- B type instruction -----
        if opCode == '1100011':
            self.state.EX['is_branch'] = 1
            self.state.EX['Operand2FromRS2'] = 1
            self.state.EX['RdDMem'] = 0
            self.state.EX['WrDMem'] = 0
            MemtoReg = 0
            regWrite = 0
            s = instruction[0] + instruction[24] + instruction[1:7] + instruction[20:24] + '0'
            imm = self.binStr2num(s, 13)
            self.state.EX['RS1_val'] = self.myRF.readRF(self.state.EX["RS1"])
            self.state.EX['RS2_val'] = self.myRF.readRF(self.state.EX["RS2"])
            if func3 == '000':  # BEQ
                self.state.EX['AluOperation'] = 6
            elif func3 == '001':  # BNE
                self.state.EX['AluOperation'] = 7

        # ----- I type instruction ----- (load)
        if opCode == '0000011':
            self.state.EX['is_branch'] = 0
            self.state.EX['Operand2FromRS2'] = 0
            self.state.EX['RdDMem'] = 1
            self.state.EX['WrDMem'] = 0
            MemtoReg = 1
            regWrite = 1
            imm = self.binStr2num(instruction[0:12], 12)
            self.state.EX['RS1_val'] = self.myRF.readRF(self.state.EX["RS1"])
            self.state.EX['AluOperation'] = 0
            # print('LW x%d %d(x%d)' %(rd,imm,rs1))
            # print(rd,imm,data,rs1_val+imm)

        # ----- S type instruction -----
        if opCode == '0100011':
            self.state.EX['is_branch'] = 0
            self.state.EX['Operand2FromRS2'] = 0
            self.state.EX['RdDMem'] = 0
            self.state.EX['WrDMem'] = 1
            MemtoReg = 0
            regWrite = 0
            imm = self.binStr2num(instruction[0:7] + instruction[20:25], 12)
            self.state.EX['RS1_val'] = self.myRF.readRF(self.state.EX["RS1"])
            self.state.EX['RS2_val'] = self.myRF.readRF(self.state.EX["RS2"])
            self.state.EX['AluOperation'] = 0

        return MemtoReg, regWrite, imm

    def ALU(self, imm):
        branch_MUX = 0
        rs1 = self.state.EX['RS1_val']
        rs2 = 0
        if self.state.EX['Operand2FromRS2'] == 1:
            rs2 = self.state.EX['RS2_val']
        else:
            rs2 = imm
        if self.state.EX['AluOperation'] == 0:
            self.state.EX['ALUresult'] = rs1 + rs2
        elif self.state.EX['AluOperation'] == 1:
            self.state.EX['ALUresult'] = rs1 - rs2
        elif self.state.EX['AluOperation'] == 2:
            self.state.EX['ALUresult'] = rs1 ^ rs2
        elif self.state.EX['AluOperation'] == 3:
            self.state.EX['ALUresult'] = rs1 | rs2
        elif self.state.EX['AluOperation'] == 4:
            self.state.EX['ALUresult'] = rs1 & rs2
        elif self.state.EX['AluOperation'] == 5:
            self.state.EX['ALUresult'] = self.state.IF['PC'] + 4
            branch_MUX = 1
        elif self.state.EX['AluOperation'] == 6:
            self.state.EX['ALUresult'] = rs1 - rs2
            if self.state.EX['ALUresult'] == 0:
                branch_MUX = 1
        elif self.state.EX['AluOperation'] == 7:
            self.state.EX['ALUresult'] = rs1 - rs2
            if self.state.EX['ALUresult'] != 0:
                branch_MUX = 1
        return branch_MUX

    def branch(self, branch_MUX, imm):
        pc1 = self.state.IF['PC'] + 4
        pc2 = self.state.IF['PC'] + imm
        if branch_MUX and self.state.EX['is_branch'] == 1:
            self.nextState.IF['PC'] = pc2
        else:
            self.nextState.IF['PC'] = pc1

    def executeData(self):
        if self.state.EX['RdDMem'] == 1:
            data = self.ext_dmem.readData(self.state.EX['ALUresult'])
            data = self.hexStr2num(data)
            self.state.MEM['rd_mem'] = data
        if self.state.EX['WrDMem'] == 1:
            address = self.state.EX['ALUresult']
            self.ext_dmem.writeDataMem(address, self.state.EX['RS2_val'])

    def write2register(self, MemtoReg, regWrite):
        if regWrite:
            if MemtoReg:
                self.myRF.writeRF(self.state.EX["DestReg"], self.state.MEM['rd_mem'])
            else:
                self.myRF.writeRF(self.state.EX["DestReg"], self.state.EX['ALUresult'])

    def step(self):
        if not self.state.IF["nop"]:
            if self.state.IF['PC'] not in self.instruLocationList:
                self.instruLocationList.append(self.state.IF['PC'])
                self.instrNumber += 1

        if self.state.IF["nop"]:
            self.halted = True
            # self.printResult(self.instrNumber, self.cycle)
        else:
            # get the the instruction
            print(self.state.IF['PC'])
            sub_instruction = self.ext_imem.readInstr(self.state.IF['PC'])
            sub_instruction = bin(int(sub_instruction, 16))[2:]
            instruction = ['0' for _ in range(32 - len(sub_instruction))]
            instruction = ''.join(instruction) + sub_instruction

            # decode the instruction
            opCode, func3, func7 = self.deCodeInstr(instruction)

            if opCode == '1111111':
                self.nextState.IF["nop"] = True
            else:
                # control module
                MemtoReg, regWrite, imm = self.control(opCode, func3, func7, instruction)

                # execute ALU
                branch_MUX = self.ALU(imm)

                # execute Branch
                self.branch(branch_MUX, imm)

                # execute data memory
                self.executeData()

                # write to register
                self.write2register(MemtoReg, regWrite)

        self.myRF.outputRF(self.cycle)  # dump RF

        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        self.state = copy.deepcopy(self.nextState)  # The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")

        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + "\\FS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_FS.txt"
        self.instrNumber=1

    def binStr2num(self, strData,n):
        if strData[0] == '0':
            data = int(strData[1:],2)
        else:
            data = int(strData[1:],2)
            data = 2**(n-1) - data
            data = -1 * data
        return data

    def hexStr2num(self, strData):
        negativeTrans = {
            '8':'0',
            '9':'1',
            'a':'2',
            'b':'3',
            'c':'4',
            'd':'5',
            'e':'6',
            'f':'7'
        }
        if strData[0] in ['0','1','2','3','4','5','6','7']:
            data = int(strData,16)
        else:
            data = negativeTrans[strData[0]] + strData[1:]
            data = int(data,16)
            data = 2**31 - data
            data = -1 * data
        return data

    def alu_execution(self, operand1, operand2, alu_control):
        result = 0
        if alu_control == 2:  # add
            result = operand1 + operand2
        elif alu_control == 6:  # subtract
            result = operand1 - operand2
        elif alu_control == 0:  # and
            result = operand1 & operand2
        elif alu_control == 1:  # or
            result = operand1 | operand2
        elif alu_control == 3:  # xor
            result = operand1 ^ operand2
        else: print('alu_control signal is illegal.')
        return result


    def step(self):
        # Your implementation
        print('========== cycle', self.cycle, '==========')

        # --------------------- WB stage ---------------------
        if self.state.WB["nop"]:  # If the program has just started
            self.nextState.WB = copy.deepcopy(self.state.WB)  # state remains
            self.nextState.WB["nop"] = 0
        else:
            if self.state.WB["wrt_enable"]:
                self.myRF.writeRF(self.state.WB["Wrt_reg_addr"], self.state.WB["Wrt_data"])

        # --------------------- MEM stage --------------------
        if self.state.MEM["nop"]:  # If the program has just started
            self.nextState.MEM = copy.deepcopy(self.state.MEM)  # state remains
            self.nextState.MEM["nop"] = 0  # state remains except nop
            self.nextState.WB["nop"] = 1  # next stage also nop
        else:
            Wrt_data = 0
            if self.state.MEM["wrt_mem"]:
                self.ext_dmem.writeDataMem(self.state.MEM["ALUresult"], self.state.MEM["Store_data"])
            elif self.state.MEM["rd_mem"]:
                data = self.ext_dmem.readData(self.state.MEM["ALUresult"])
                Wrt_data = self.hexStr2num(data)
            else: Wrt_data = self.state.MEM["ALUresult"]

            self.nextState.WB["nop"] = 0
            self.nextState.WB["Wrt_data"] = Wrt_data
            self.nextState.WB["Wrt_reg_addr"] = self.state.MEM["Wrt_reg_addr"]
            self.nextState.WB["wrt_enable"] = self.state.MEM["wrt_enable"]

        # --------------------- EX stage ---------------------
        print('EX stage',)
        if self.state.EX["nop"]:  # If the program has just started
            self.nextState.EX = copy.deepcopy(self.state.EX)  # state remains
            self.nextState.EX["nop"] = 0  # state remains except nop
            self.nextState.MEM["nop"] = 1  # next stage also nop

        else:
            # Forward detection and implementation
            if self.state.EX["RS1"] == self.state.MEM["Wrt_reg_addr"] and self.state.MEM["wrt_enable"] and not self.state.MEM["nop"]:
                self.state.EX["RS1_val"] = self.state.MEM["ALUresult"]
            elif self.state.EX["RS1"] == self.state.WB["Wrt_reg_addr"] and self.state.WB["wrt_enable"] and not self.state.WB["nop"]:
                self.state.EX["RS1_val"] = self.state.WB["Wrt_data"]
            if self.state.EX["RS2"] == self.state.MEM["Wrt_reg_addr"] and self.state.MEM["wrt_enable"] and not self.state.MEM["nop"]:
                self.state.EX["RS2_val"] = self.state.MEM["ALUresult"]
            elif self.state.EX["RS2"] == self.state.WB["Wrt_reg_addr"] and self.state.WB["wrt_enable"] and not self.state.WB["nop"]:
                self.state.EX["RS2_val"] = self.state.WB["Wrt_data"]


            alu_result = 0
            if self.state.EX["is_branch"]:
                alu_result = self.alu_execution(self.state.EX["RS1_val"], self.state.EX["RS2_val"],
                                                self.state.EX["AluOperation"])
                if (self.state.EX["AluOperation"] == 6 and not alu_result) or (self.state.EX["AluOperation"] == 3 and alu_result):

                    self.nextState.IF["PC"] = self.alu_execution(self.state.IF["PC"]-8, self.state.EX["imm"], 2)
                    self.nextState.ID["nop"] = 1  # branch taken, stop for two cycles
                    self.nextState.EX["nop"] = 1  # branch taken, stop for two cycles

            elif self.state.EX["Operand2FromRS2"]:
                alu_result = self.alu_execution(self.state.EX["RS1_val"], self.state.EX["RS2_val"],
                                                self.state.EX["AluOperation"])

            elif not self.state.EX["Operand2FromRS2"]:
                alu_result = self.alu_execution(self.state.EX["RS1_val"], self.state.EX["imm"],
                                                self.state.EX["AluOperation"])

            self.nextState.MEM["nop"] = 0
            self.nextState.MEM["ALUresult"] = alu_result

            self.nextState.MEM["Store_data"] = self.state.EX["RS2_val"]
            self.nextState.MEM["Wrt_reg_addr"] = self.state.EX["DestReg"]
            self.nextState.MEM["rd_mem"] = self.state.EX["RdDMem"]
            self.nextState.MEM["wrt_mem"] = self.state.EX["WrDMem"]
            self.nextState.MEM["wrt_enable"] = self.state.EX["WBEnable"]

        # --------------------- ID stage ---------------------

        if self.state.ID["nop"]:  # If the program has just started
            self.nextState.ID = copy.deepcopy(self.state.ID)  # state remains
            self.nextState.EX["nop"] = 1  # next stage also nop
        else:
            opCode = self.state.ID['Instr'][25:]
            print('ID stage: opcode',opCode)
            # ----- R type instruction -----
            if opCode == '0110011':

                func3 = self.state.ID['Instr'][17:20]
                func7 = self.state.ID['Instr'][0:7]
                rd = int(self.state.ID['Instr'][20:25], 2)
                rs1 = int(self.state.ID['Instr'][12:17], 2)
                rs2 = int(self.state.ID['Instr'][7:12], 2)
                rs1_val = self.myRF.readRF(rs1)
                rs2_val = self.myRF.readRF(rs2)

                self.nextState.EX["nop"] = 0
                self.nextState.EX["RS1"] = rs1
                self.nextState.EX["RS2"] = rs2
                self.nextState.EX["RS1_val"] = rs1_val
                self.nextState.EX["RS2_val"] = rs2_val
                self.nextState.EX["imm"] = 0
                self.nextState.EX["Operand2FromRS2"] = 1
                self.nextState.EX["DestReg"] = rd
                self.nextState.EX["AluOperation"] = None
                self.nextState.EX["is_branch"] = 0
                self.nextState.EX["WBEnable"] = 1
                self.nextState.EX["RdDMem"] = 0
                self.nextState.EX["WrDMem"] = 0
                self.nextState.EX["Halt"] = 0

                if func3 == '000':
                    if func7 == '0000000':  # ADD
                        self.nextState.EX["AluOperation"] = 2  # add option - 0010
                    else:  # SUB
                        self.nextState.EX["AluOperation"] = 6  # subtract option - 0110
                elif func3 == '100':  # XOR
                    self.nextState.EX["AluOperation"] = 3  # xor option - 0011 (self-defined)
                elif func3 == '110':  # OR
                    self.nextState.EX["AluOperation"] = 1  # or option - 0001
                elif func3 == '111':  # AND
                    self.nextState.EX["AluOperation"] = 0  # and option - 0000

            # ----- I type instruction ----- (normal)
            elif opCode == '0010011':
                func3 = self.state.ID['Instr'][17:20]
                rd = int(self.state.ID['Instr'][20:25], 2)
                rs1 = int(self.state.ID['Instr'][12:17], 2)
                rs1_val = self.myRF.readRF(rs1)
                imm = self.binStr2num(self.state.ID['Instr'][0:12], 12)

                self.nextState.EX["nop"] = 0
                self.nextState.EX["RS1"] = rs1
                self.nextState.EX["RS2"] = 0
                self.nextState.EX["RS1_val"] = rs1_val
                self.nextState.EX["RS2_val"] = 0
                self.nextState.EX["imm"] = imm
                self.nextState.EX["Operand2FromRS2"] = 0
                self.nextState.EX["DestReg"] = rd
                self.nextState.EX["AluOperation"] = None
                self.nextState.EX["is_branch"] = 0
                self.nextState.EX["WBEnable"] = 1
                self.nextState.EX["RdDMem"] = 0
                self.nextState.EX["WrDMem"] = 0
                self.nextState.EX["Halt"] = 0

                if func3 == '000':  # ADDI
                    self.nextState.EX["AluOperation"] = 2  # add option - 0010
                if func3 == '100':  # XORI
                    self.nextState.EX["AluOperation"] = 3  # xor option - 0011 (self-defined)
                if func3 == '110':  # ORI
                    self.nextState.EX["AluOperation"] = 1  # or option - 0001
                if func3 == '111':  # ANDI
                    self.nextState.EX["AluOperation"] = 0  # and option - 0000

            # ----- J type instruction -----
            elif opCode == '1101111':
                rd = int(self.state.ID['Instr'][20:25], 2)
                s = self.state.ID['Instr'][0] + self.state.ID['Instr'][12:20] + self.state.ID['Instr'][11] + self.state.ID['Instr'][1:11] + '0'
                imm = self.binStr2num(s, 21)

                self.nextState.EX["nop"] = 0
                self.nextState.EX["RS1"] = 0
                self.nextState.EX["RS2"] = 0
                self.nextState.EX["RS1_val"] = self.state.IF['PC']
                self.nextState.EX["RS2_val"] = 0
                self.nextState.EX["imm"] = imm
                self.nextState.EX["Operand2FromRS2"] = 1
                self.nextState.EX["DestReg"] = rd
                self.nextState.EX["AluOperation"] = 2
                self.nextState.EX["is_branch"] = 0
                self.nextState.EX["WBEnable"] = 0
                self.nextState.EX["RdDMem"] = 0
                self.nextState.EX["WrDMem"] = 0
                self.nextState.EX["Halt"] = 0

                if not self.nextState.IF["PC"] != self.state.IF["PC"]:  # branch not taken
                    self.nextState.IF['PC'] = self.state.IF['PC']-4 + imm
                    # self.nextState.IF['PC'] = self.state.IF['PC']
                    self.nextState.ID['nop'] = 1
                    self.nextState.EX["WBEnable"] = 1

            # ----- B type instruction -----
            elif opCode == '1100011':
                func3 = self.state.ID['Instr'][17:20]
                rs1 = int(self.state.ID['Instr'][12:17], 2)
                rs2 = int(self.state.ID['Instr'][7:12], 2)
                rs1_val = self.myRF.readRF(rs1)
                rs2_val = self.myRF.readRF(rs2)
                s = self.state.ID['Instr'][0] + self.state.ID['Instr'][24] + self.state.ID['Instr'][1:7] + self.state.ID['Instr'][20:24] + '0'
                imm = self.binStr2num(s, 13)

                if rs1 == self.nextState.MEM["Wrt_reg_addr"] and self.nextState.MEM["wrt_enable"] and not self.nextState.MEM["nop"]:
                    rs1_val = self.nextState.MEM["ALUresult"]
                elif rs1 == self.nextState.WB["Wrt_reg_addr"] and self.nextState.WB["wrt_enable"] and not self.nextState.WB["nop"]:
                    rs1_val = self.nextState.WB["Wrt_data"]
                if rs2 == self.nextState.MEM["Wrt_reg_addr"] and self.nextState.MEM["wrt_enable"] and not self.nextState.MEM["nop"]:
                    rs2_val = self.nextState.MEM["ALUresult"]
                elif rs2 == self.nextState.WB["Wrt_reg_addr"] and self.nextState.WB["wrt_enable"] and not self.nextState.WB["nop"]:
                    rs2_val = self.nextState.WB["Wrt_data"]

                self.nextState.EX["nop"] = 1
                self.nextState.EX["RS1"] = rs1
                self.nextState.EX["RS2"] = rs2
                self.nextState.EX["RS1_val"] = rs1_val
                self.nextState.EX["RS2_val"] = rs2_val
                self.nextState.EX["imm"] = imm
                self.nextState.EX["Operand2FromRS2"] = 1
                self.nextState.EX["DestReg"] = 0
                self.nextState.EX["AluOperation"] = None
                self.nextState.EX["is_branch"] = 1
                self.nextState.EX["WBEnable"] = 0
                self.nextState.EX["RdDMem"] = 0
                self.nextState.EX["WrDMem"] = 0
                self.nextState.EX["Halt"] = 0

                alu_option = 6
                if func3 == '000':  # BEQ
                    alu_option = 6  # subtract option - 0110
                elif func3 == '001':  # BNE
                    alu_option = 3  # xor option - 0011
                alu_result = self.alu_execution(rs1_val, rs2_val, alu_option)

                if (alu_option == 6 and not alu_result) or (alu_option == 3 and alu_result):
                    self.nextState.IF["PC"] = self.alu_execution(self.state.IF["PC"] - 4, imm, 2)
                    self.nextState.ID["nop"] = 1  # branch taken, stop for two cycles
                    self.nextState.EX["nop"] = 1  # branch taken, stop for two cycles

            # ----- I type instruction ----- (load)
            elif opCode == '0000011':
                rd = int(self.state.ID['Instr'][20:25], 2)
                rs1 = int(self.state.ID['Instr'][12:17], 2)
                rs1_val = self.myRF.readRF(rs1)
                imm = self.binStr2num(self.state.ID['Instr'][0:12], 12)

                self.nextState.EX["nop"] = 0
                self.nextState.EX["RS1"] = rs1
                self.nextState.EX["RS2"] = 0
                self.nextState.EX["RS1_val"] = rs1_val
                self.nextState.EX["RS2_val"] = 0
                self.nextState.EX["imm"] = imm
                self.nextState.EX["Operand2FromRS2"] = 0
                self.nextState.EX["DestReg"] = rd
                self.nextState.EX["AluOperation"] = 2  # add option - 0010
                self.nextState.EX["is_branch"] = 0
                self.nextState.EX["WBEnable"] = 1
                self.nextState.EX["RdDMem"] = 1
                self.nextState.EX["WrDMem"] = 0
                self.nextState.EX["Halt"] = 0

            # ----- S type instruction -----
            elif opCode == '0100011':
                # print(self.ext_dmem.DMem)
                rs1 = int(self.state.ID['Instr'][12:17], 2)
                rs2 = int(self.state.ID['Instr'][7:12], 2)
                rs1_val = self.myRF.readRF(rs1)
                rs2_val = self.myRF.readRF(rs2)
                imm = self.binStr2num(self.state.ID['Instr'][0:7] + self.state.ID['Instr'][20:25], 12)

                self.nextState.EX["nop"] = 0
                self.nextState.EX["RS1"] = rs1
                self.nextState.EX["RS2"] = rs2
                self.nextState.EX["RS1_val"] = rs1_val
                self.nextState.EX["RS2_val"] = rs2_val
                self.nextState.EX["imm"] = imm
                self.nextState.EX["Operand2FromRS2"] = 0
                self.nextState.EX["DestReg"] = 0
                self.nextState.EX["AluOperation"] = 2  # add option - 0010
                self.nextState.EX["is_branch"] = 0
                self.nextState.EX["WBEnable"] = 0
                self.nextState.EX["RdDMem"] = 0
                self.nextState.EX["WrDMem"] = 1
                self.nextState.EX["Halt"] = 0

            # ----- halt instruction -----
            # else:
            #     self.nextState.ID['nop'] = 1
            #
            #     self.nextState.EX['nop'] = 1
            #     if not self.nextState.IF["PC"] != self.state.IF["PC"]:  # branch not taken
            #         self.halt = 1
            #         self.nextState.IF['nop'] = 1
            #         self.nextState.ID['nop'] = 1

            # ----- load-use hazard  -----
            if ((self.nextState.MEM["rd_mem"] and self.nextState.EX["RS1"] == self.nextState.MEM["Wrt_reg_addr"] and not self.nextState.MEM["nop"])
                or (self.nextState.MEM["rd_mem"] and self.nextState.EX["RS2"] == self.nextState.MEM["Wrt_reg_addr"] and not self.nextState.MEM["nop"])):  # load-use hazard
                self.state.IF["nop"] = 1
                self.nextState.EX["nop"] = 1  # next stage also nop
            elif not self.halt:
                pass
            else:  # if halt
                self.nextState.ID["nop"] = 1
                self.nextState.EX["nop"] = 1

        # --------------------- IF stage ---------------------

        if self.halt:
            self.state.IF["nop"] = 1

        sub_instruction = self.ext_imem.readInstr(self.state.IF['PC'])
        sub_instruction = bin(int(sub_instruction, 16))[2:]
        instruction = ['0' for _ in range(32 - len(sub_instruction))]
        instruction = ''.join(instruction) + sub_instruction
        print('IF stage',instruction ,'PC',self.state.IF['PC'])

        if instruction == '11111111111111111111111111111111' and self.nextState.IF["PC"] == self.state.IF["PC"]:
            self.halt = 1
            self.state.IF["nop"] = 1
            self.nextState.ID["nop"] = 1

        if self.nextState.IF['PC'] == self.state.IF['PC'] and not self.state.IF['nop'] and not self.halt:
            # if no jump or branch is calculated in other stages and to be taken, or there is a load-use hazard
            self.nextState.IF['PC'] = self.state.IF['PC'] + 4
            self.nextState.ID['nop'] = 0
            self.nextState.ID['Instr'] = instruction


        # self.nextState.ID["nop"] = 0


        # ----- Five stages Done -----

        # self.halted = True
        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.halted = True
        print(self.state.IF["nop"],self.state.ID["nop"],self.state.EX["nop"],self.state.MEM["nop"],self.state.WB["nop"])
        
        self.myRF.outputRF(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        # if self.state.IF['PC'] != self.nextState.IF['PC'] and not self.state.IF['nop']:
        #     self.instrNumber += 1

        self.state = copy.deepcopy(self.nextState)  # The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

        print("halt",self.halt)

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


def output_PerformanceMetrics_Result(core1, core2, ioDir):

    resPath = ioDir + "\\" + "PerformanceMetrics_Result.txt"
    CPIss = core1.cycle/core1.instrNumber
    CPIfs = core2.cycle / core1.instrNumber

    with open(resPath, "w") as rp:
        rp.writelines("IO Directory: "+ ioDir + '\n')
        rp.writelines("Single Stage Core Performance Metrics-----------------------------"+'\n')
        rp.writelines("Number of cycles taken: " + str(core1.cycle)+'\n')
        rp.writelines("Cycles per instruction: : " + str(CPIss)+'\n')
        rp.writelines("Instructions per cycle: : " + str(1/CPIss) + '\n' + '\n')

        rp.writelines("Five Stage Core Performance Metrics-----------------------------"+'\n')
        rp.writelines("Number of cycles taken: " + str(core2.cycle)+'\n')
        rp.writelines("Cycles per instruction: : " + str(CPIfs)+'\n')
        rp.writelines("Instructions per cycle: : " + str(1/CPIfs) + '\n')


if __name__ == "__main__":
    #parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    ioDir = os.path.abspath(args.iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)
    
    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while(True):
        if not ssCore.halted:
            ssCore.step()
        
        if not fsCore.halted:
            fsCore.step()
        if ssCore.halted and fsCore.halted:
            break

    output_PerformanceMetrics_Result(ssCore, fsCore, ioDir)
    
    # dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()