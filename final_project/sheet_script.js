//------------------------------------------------------------------------------------
// Attendance and Registration Mode.
//________________________________________________________________________________doGet()
function doGet(e) {
  Logger.log(JSON.stringify(e));
  var result = 'OK';
  if (e.parameter == 'undefined') {
    result = 'No_Parameters';
  }
  else {
    var sheet_id = '1J4fqWgBNGXjfPLxFnxt8zv2sEo1DqzbcGKcKqBRGFQ8';  // Spreadsheet ID.
    var sheet_UD = 'ID2Name';  
    var sheet_AT = 'Raw_Reciepts';  


    var sheet_open = SpreadsheetApp.openById(sheet_id);
    var sheet_user_data = sheet_open.getSheetByName(sheet_UD);
    var sheet_attendence = sheet_open.getSheetByName(sheet_AT);
   
    var sts_val = "";
    var uid_val = "";
    var uid_column = "A";
    var CI_val = "";
    var Date_val = "";
   
    for (var param in e.parameter) {
      var value = stripQuotes(e.parameter[param]);
      switch (param) {
        case 'sts':
          sts_val = value;
          break;
        case 'uid':
          uid_val = value;
          break;
      }
    }
   
    //----------------- Register new user
    if (sts_val == 'reg') {
      var check_new_UID = checkUID(sheet_id, sheet_UD, 1, uid_val);
      if (check_new_UID == true) {
        result += ",regErr01";
        return ContentService.createTextOutput(result);
      }
      var getLastRowUIDCol = findLastRow(sheet_id, sheet_UD, uid_column);  
      var newUID = sheet_open.getRange(uid_column + (getLastRowUIDCol + 1));
      newUID.setValue(uid_val);
      result += ",R_Successful";
      return ContentService.createTextOutput(result);
    }


    //----------------- Attendance mode
    if (sts_val == 'atc') {
      var FUID = findUID(sheet_id, sheet_UD, 1, uid_val);
      if (FUID == -1) {
        result += ",atcErr01";
        return ContentService.createTextOutput(result);
      } else {
        var get_Range = sheet_user_data.getRange("B" + (FUID+2));
        var user_name_by_UID = get_Range.getValue();


        var enter_data = "check_in";
        var num_row;
        var Curr_Date = Utilities.formatDate(new Date(), "America/Chicago", 'MM/dd/yyyy');
        var Curr_Time = Utilities.formatDate(new Date(), "America/Chicago", 'HH:mm:ss');


        var data = sheet_attendence.getDataRange().getDisplayValues();
       
        if (data.length > 1) {
          for (var i = 0; i < data.length; i++) {
            if (data[i][0] == uid_val) {
              if (data[i][1] == Curr_Date) {
                if (data[i][3] == "") {
                  Date_val = data[i][1];
                  CI_val = data[i][2];
                  enter_data = "check_out";
                  num_row = i + 1;
                  break;
                } else {
                  Date_val = data[i][1];
                  CI_val = data[i][2];
                  enter_data = "update_checkout";
                  num_row = i + 1;
                  break;
                }
              }
            }
          }
        }
       
        //----------------- Check in
        if (enter_data == "check_in") {
          sheet_attendence.insertRows(2);
          sheet_attendence.getRange("A2").setValue(uid_val);
          sheet_attendence.getRange("B2").setValue(Curr_Date);
          sheet_attendence.getRange("C2").setValue(Curr_Time);
          SpreadsheetApp.flush();
          result += ",CI_Successful" + "," + user_name_by_UID + "," + Curr_Date + "," + Curr_Time;
          return ContentService.createTextOutput(result);
        }
       
        //----------------- Check out (calculate hours)
        if (enter_data == "check_out" || enter_data == "update_checkout") {
          sheet_attendence.getRange("D" + num_row).setValue(Curr_Time);


          // Calculate hours between CI and CO
          var ciDateTime = new Date(Date_val + " " + CI_val);
          var coDateTime = new Date(Date_val + " " + Curr_Time);
          var diffMs = coDateTime - ciDateTime;
          var diffHours = diffMs / (1000 * 60 * 60); // convert ms to hours
          sheet_attendence.getRange("E" + num_row).setValue(diffHours);


          result += (enter_data == "check_out" ? ",CO_Successful" : ",CO_Updated")
                    + "," + user_name_by_UID + "," + Date_val + "," + CI_val + "," + Curr_Time + "," + diffHours;
          return ContentService.createTextOutput(result);
        }
      }
    }
  }
}
//________________________________________________________________________________
function stripQuotes(value) {
  return value.replace(/^["']|['"]$/g, "");
}
function findLastRow(id_sheet, name_sheet, name_column) {
  var spreadsheet = SpreadsheetApp.openById(id_sheet);
  var sheet = spreadsheet.getSheetByName(name_sheet);
  var lastRow = sheet.getLastRow();
  var range = sheet.getRange(name_column + lastRow);
  if (range.getValue() !== "") {
    return lastRow;
  } else {
    return range.getNextDataCell(SpreadsheetApp.Direction.UP).getRow();
  }
}
function findUID(id_sheet, name_sheet, column_index, searchString) {
  var open_sheet = SpreadsheetApp.openById(id_sheet);
  var sheet = open_sheet.getSheetByName(name_sheet);
  var columnValues = sheet.getRange(2, column_index, sheet.getLastRow()).getValues();
  var searchResult = columnValues.findIndex(searchString);
  return searchResult;
}
function checkUID(id_sheet, name_sheet, column_index, searchString) {
  var open_sheet = SpreadsheetApp.openById(id_sheet);
  var sheet = open_sheet.getSheetByName(name_sheet);
  var columnValues = sheet.getRange(2, column_index, sheet.getLastRow()).getValues();
  var searchResult = columnValues.findIndex(searchString);
  if(searchResult != -1) {
    //sheet.setActiveRange(sheet.getRange(searchResult + 2, 2)).setValue("UID has been registered in this row.");
    return true;
  } else {
    return false;
  }
}
Array.prototype.findIndex = function(search){
  if(search == "") return false;
  for (var i=0; i<this.length; i++)
    if (this[i].toString().indexOf(search) > -1 ) return i;
  return -1;
}
