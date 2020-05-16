from pyexcelerate import Workbook,Style
import datetime as dt

def makeExcel(result):
    wb = Workbook()

    for i in range(len(result[0])):
        GAME = result[0][i][0]
        MUTATION_RATE = result[0][i][1]
        LAYERS = result[0][i][2]
        MAX_GENERATIONS = result[0][i][3]
        POPULATION_COUNT = result[0][i][4]
        ELAPSED_TIME = result[0][i][5]
        COL_TOTALS = result[0][i][6]
        INFO = result[0][i][7]
        GENERATIONS = result[0][i][8]


        sheetname = f"Genetics_{i+1}_{MUTATION_RATE}"
        ws = wb.new_sheet(f"{i+1}")
        for i in range(6):
            ws.set_col_style(i+1,Style(size=-1))
        ws.cell("A1").value = GAME
        ws.cell("A2").value = "Genetic Alg"
        ws.cell("A3").value = sheetname
        ws.cell("B1").value = "Mutation rate:"
        ws.cell("B2").value = MUTATION_RATE
        ws.cell("C1").value = "Generations:"
        ws.cell("C2").value = MAX_GENERATIONS
        ws.cell("D1").value = "Population count:"
        ws.cell("D2").value = POPULATION_COUNT
        ws.cell("E1").value = "Hidden Layers:"
        lay = ''
        for i in range(len(LAYERS)):
            lay += f"{LAYERS[i]}, "
        ws.cell("E2").value = lay
        ws.cell("F1").value = "Time(ms):"
        ws.cell("F2").value = ELAPSED_TIME
        ws.cell('G1').value = INFO

        startrow = 4
        startcol = 2
        ws[startrow - 1][startcol].value = "Generation"
        ws[startrow - 1][startcol + 1].value = "Average Fitness"
        ws[startrow - 1][startcol + 2].value = "Max Fitness"
        for i in range(len(GENERATIONS)):
            for j in range(len(GENERATIONS[0])):
                ws[startrow + i][startcol + j].value = int(GENERATIONS[i][j])
        ws[startrow + len(GENERATIONS)][startcol].value = "Total:"
        ws[startrow + len(GENERATIONS)][startcol + 1] = int(COL_TOTALS[1])
        ws[startrow + len(GENERATIONS)][startcol + 2] = int(COL_TOTALS[2])

    wb.save(f"C:\\Users\\dosha\\Desktop\\ExcelFiles\\GA_{GAME}_{dt.datetime.now().strftime('%f')}.xlsx")
    print("..........Excel file generated..............")