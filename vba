Sub CalculateAttendancePercentage()
    Dim wsData As Worksheet
    Dim wsOutput As Worksheet
    
    ' Set the data and output worksheets
    Set wsData = ThisWorkbook.Sheets("Sheet1") ' Replace "Sheet1" with your data sheet name
    Set wsOutput = ThisWorkbook.Sheets("Sheet2") ' Replace "Sheet2" with your output sheet name
    
    Dim lastRow As Long
    lastRow = wsData.Cells(wsData.Rows.Count, "A").End(xlUp).Row
    
    Dim totalWorkingDays As Long
    totalWorkingDays = 0
    
    Dim totalHolidaysApplied As Long
    totalHolidaysApplied = 0
    
    Dim totalAttendances As Long
    totalAttendances = 0
    
    Dim i As Long
    For i = lastRow To 2 Step -1 ' Assuming your data starts from row 2
        If i >= (lastRow - 83) Then ' Calculate for the last 12 weeks (84 days)
            If wsData.Cells(i, "B").Value = "No" Then ' Check if it was not a holiday
                totalWorkingDays = totalWorkingDays + 1
                totalAttendances = totalAttendances + wsData.Cells(i, "C").Value
                totalHolidaysApplied = totalHolidaysApplied + wsData.Cells(i, "D").Value
            End If
        Else
            Exit For ' Exit the loop once you have considered 12 weeks of data
        End If
    Next i
    
    If totalWorkingDays - totalHolidaysApplied > 0 Then
        Dim attendancePercentage As Double
        attendancePercentage = (totalAttendances / (totalWorkingDays - totalHolidaysApplied)) * 100
        
        ' Output the result in Sheet2
        wsOutput.Cells(2, 2).Value = "Attendance Percentage for the last 12 weeks:"
        wsOutput.Cells(2, 3).Value = Format(attendancePercentage, "0.00") & "%"
    Else
        wsOutput.Cells(2, 2).Value = "No attendance data available for the last 12 weeks."
    End If
End Sub
