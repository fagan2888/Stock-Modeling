library(shiny)
library(dygraphs)
library(xts)
library(DT)
library(quantmod)
library(shinydashboard)
library(data.table)

#### Setup Data Source References #############################################################################
DJ_NEWS_SENTIMENT_DATA <- fread("https://raw.githubusercontent.com/mwilchek/GWU-Data-Mining-Proposal-1/master/DJ_NEWS_SENTIMENT_DATA.csv")
predicted_data <- fread("https://raw.githubusercontent.com/mwilchek/GWU-Data-Mining-Proposal-1/master/predictions_djia.csv")
djia <- data.frame(DJ_NEWS_SENTIMENT_DATA)
predictions <- data.frame(predicted_data)
predictions_show <- data.frame(predicted_data)
last_prediction <- tail(predictions, 1)
rownames(djia) <- djia$Date
rownames(predictions) <- predictions$Date
predictions$Date <- NULL
djia$Date <- NULL
djia <- xts(djia, order.by=as.Date(rownames(djia),"%m/%d/%Y"))
predictions <- xts(predictions, order.by=as.Date(rownames(predictions),"%m/%d/%Y"))
today_record <- tail(djia, 1)

#### Placeholders for Left Navigation and Associated Icons #####################################################
sidebar <-
  dashboardSidebar(
    sidebarMenu(
      menuItem(
        "Closing Values",
        tabName = "closing",
        icon = icon("bar-chart-o")
      ),
      menuItem(
        "Low Values",
        tabName = "low",
        icon = icon("bar-chart-o")
      ),
      menuItem(
        "High Values",
        tabName = "high",
        icon = icon("bar-chart-o")
      ),
      menuItem(
        "About Dashboard",
        tabName = "about",
        icon = icon("dashboard")
      )
    )
  )

#### Content Placeholders for Graphics and Tables for All Tabs ##################################################
body <- dashboardBody(
  tabItems(
    # First tab content
    tabItem(tabName = "closing",
            
            fluidRow(
              
              box(title='Closing Price of DJI', width = 12, height = NULL, dygraphOutput("plot1")),
              box("Note: There is a gap in available data from 7/2/2016 to 11/19/2017", width = 12, height = NULL),
              valueBox(value=round(as.double(last_prediction$Predicted.Close.Tomorrow), digits=2),subtitle="Next Business Day Predicted Value",color = 'green',icon=icon("dollar")),
              infoBox(title=NULL,value = round(as.double(last_prediction$Predicted.Close.Today), digits=2), subtitle = paste("Today Predicted Value (", toString(index(today_record)), ")"),color = 'blue',icon=icon("dollar")),
              infoBox(title=NULL,value = round(as.double(today_record$Close), digits=2), subtitle = paste("Actual Value (", toString(index(today_record)), ")"),color = 'yellow',icon=icon("dollar"))
            ),
            
            fluidRow(
              h2("Historical Predictions"),
              DT::dataTableOutput("closeTable")
            )
            
    ),
    # Second tab content
    tabItem(tabName = "low",
            
            fluidRow(
              
              box(title='Lowest Price of DJI', width = 12, height = NULL, dygraphOutput("plot2")),
              box("Note: There is a gap in available data from 7/2/2016 to 11/19/2017", width = 12, height = NULL),
              valueBox(value=round(as.double(last_prediction$Predicted.Low.Tomorrow), digits=2),subtitle="Next Day Predicted Value",color = 'green',icon=icon("dollar")),
              infoBox(title=NULL,value = round(as.double(last_prediction$Predicted.Low.Today), digits=2), subtitle = paste("Today Predicted Value (", toString(index(today_record)), ")"),color = 'blue',icon=icon("dollar")),
              infoBox(title=NULL,value = round(as.double(today_record$Low), digits=2), subtitle = paste("Actual Value (", toString(index(today_record)), ")"),color = 'yellow',icon=icon("dollar"))
            ),
            
            fluidRow(
              h2("Historical Predictions"),
              DT::dataTableOutput("lowTable")
            )
            
    ),
    # Third tab content
    tabItem(tabName = "high",
            
            fluidRow(
              
              box(title='Highest Price of DJI', width = 12, height = NULL, dygraphOutput("plot3")),
              box("Note: There is a gap in available data from 7/2/2016 to 11/19/2017", width = 12, height = NULL),
              valueBox(value=round(as.double(last_prediction$Predicted.High.Tomorrow), digits=2),subtitle="Next Day Predicted Value",color = 'green',icon=icon("dollar")),
              infoBox(title=NULL,value = round(as.double(last_prediction$Predicted.High.Today), digits=2), subtitle = paste("Today Predicted Value (", toString(index(today_record)), ")"),color = 'blue',icon=icon("dollar")),
              infoBox(title=NULL,value = round(as.double(today_record$High), digits=2), subtitle = paste("Actual Value (", toString(index(today_record)), ")"),color = 'yellow',icon=icon("dollar"))
            ),
            
            fluidRow(
              h2("Historical Predictions"),
              DT::dataTableOutput("highTable")
            )
            
    ),
    # Fourth tab content
    tabItem(tabName = "about",
            
            fluidRow(
              box("My name is Matt. I'm a graduate student at George Washington University in Data Science that enjoys being creative for data visualization. 
                  I made this dashboard to practice R Shiny Dashboards and show off the prediction results of a python program my project partner and I made. 
                  Our program collects stock data, and runs a sentiment algorithm over related news data for it. Then uses RandomForest to help create a train/test
                  model to predict the open, high, and low values for the current and following day of that specified stock.", width = 12, height = NULL),
              uiOutput("about")
            )
    )
  
  )
)

ui <- dashboardPage(dashboardHeader(title = 'Stock Predictions'), sidebar,body)

server <- shinyServer(function(input, output) {
  
#### Tables and Graphs Output for Tab 1 ##########################################################################  
  # Get Graph Timeline of Closing Values
  output$plot1 <- renderDygraph({
    
    # djia["DJI.Adjusted"] <- round(djia$Close, digits = 2)
    close <- djia$Close
    predicted_close <- predictions$Predicted.Close.Tomorrow
    stock_close <- cbind(close, predicted_close)
    
    dygraph(stock_close) %>%
      dySeries("Close", label = "Actual Close") %>%
      dySeries("Predicted.Close.Tomorrow", label = "Predicted Close") %>%
      dyOptions(stackedGraph = FALSE) %>%
      dyRangeSelector()
  })
  
  output$closeTable = DT::renderDataTable({
    predictions_show[,c("Date","Actual.Close.Today","Predicted.Close.Today", "Predicted.Close.Tomorrow")]
  })
  
#### Tables and Graphs Output for Tab 2 ##########################################################################  
  # Get Graph Timeline of Low Values
  output$plot2 <- renderDygraph({
    
    low <- djia$Low
    predicted_low <- predictions$Predicted.Low.Tomorrow
    stock_low <- cbind(low, predicted_low)
    
    dygraph(stock_low) %>%
      dySeries("Low", label = "Actual Low") %>%
      dySeries("Predicted.Low.Tomorrow", label = "Predicted Low") %>%
      dyOptions(stackedGraph = FALSE) %>%
      dyRangeSelector()
  })
  
  output$lowTable = DT::renderDataTable({
    predictions_show[,c("Date","Actual.Low.Today","Predicted.Low.Today", "Predicted.Low.Tomorrow")]
  })

#### Tables and Graphs Output for Tab 3 ##########################################################################
  # Get Graph Timeline of High Values
  output$plot3 <- renderDygraph({
    
    high <- djia$High
    predicted_high <- predictions$Predicted.High.Tomorrow
    stock_high <- cbind(high, predicted_high)
    
    dygraph(stock_high) %>%
      dySeries("High", label = "Actual High") %>%
      dySeries("Predicted.High.Tomorrow", label = "Predicted High") %>%
      dyOptions(stackedGraph = FALSE) %>%
      dyRangeSelector()
  })
  
  output$highTable = DT::renderDataTable({
    predictions_show[,c("Date","Actual.High.Today","Predicted.High.Today", "Predicted.High.Tomorrow")]
  })
  
#### Tables and Graphs Output for Tab 4 ##########################################################################
  
  url <- a("https://github.com/mwilchek/GWU-Data-Mining-Proposal-1", href="https://github.com/mwilchek/GWU-Data-Mining-Proposal-1")
  output$about <- renderUI({
    
    tagList("To learn more about the project or use our algorithm, go here:", url)
  }) 
  
})

shinyApp(ui = ui, server = server)  # executes app