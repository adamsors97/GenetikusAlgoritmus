from pyexcelerate import Workbook
import datetime as dt
def makeExcel(gens,MUTATION_RATE,GAME,MAX_GENERATIONS,POPULATION_COUNT,ELAPSED_TIME,COL_TOTALS,INFO):
    wb = Workbook()
    ws = wb.new_sheet(f"Genetics_{MUTATION_RATE}")
    ws.cell("A1").value = GAME
    ws.cell("A2").value = "Genetic Alg"
    ws.cell("B1").value = "Mutation rate:"
    ws.cell("B2").value = MUTATION_RATE
    ws.cell("C1").value = "Generations:"
    ws.cell("C2").value = MAX_GENERATIONS
    ws.cell("D1").value = "Population count:"
    ws.cell("D2").value = POPULATION_COUNT
    ws.cell("E1").value = "Time(ms):"
    ws.cell("E2").value = ELAPSED_TIME
    startrow = 4
    startcol = 2
    ws[startrow - 1][startcol].value = "Generation"
    ws[startrow - 1][startcol + 1].value = "Average Fitness"
    ws[startrow - 1][startcol + 2].value = "Max Fitness"
    for i in range(len(gens)):
        for j in range(len(gens[0])):
            ws[startrow + i][startcol + j].value = gens[i][j]
    ws[startrow + len(gens)][startcol + 1] = COL_TOTALS[1]
    ws[startrow + len(gens)][startcol + 2] = COL_TOTALS[2]

    wb.save(f"C:\\Users\\dosha\\Desktop\\ExcelFiles\\GA_{GAME}_{dt.datetime.now().strftime('%f')}_{INFO}.xlsx")
    print("..........Excel file generated..............")