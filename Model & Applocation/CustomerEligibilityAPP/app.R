# app.R
library(shiny)
library(randomForest)
library(DT)
library(dplyr)
library(ggplot2)
library(caret) # For accuracy and cross-validation

# Load your dataset
data_path <-"~/Belgium Campus/BIN 381/Milestone 7/Prepared_Data_Cleaned.csv"
prepared_data <- read.csv(data_path)

# Set eligibility thresholds
salary_threshold <- 50000
ytd_threshold <- 45000
residence_threshold <- 2.5

# Ensure relevant columns exist in the dataset
if (!all(c("Annual.Salary", "Gross_Year_To_Date", "yrs_residence") %in% names(prepared_data))) {
  stop("Dataset does not contain required columns: Annual.Salary, Gross_Year_To_Date, and yrs_residence")
}

# Prepare eligibility column based on the thresholds
prepared_data$Eligible <- factor(
  ifelse(prepared_data$Annual.Salary >= salary_threshold & 
           prepared_data$Gross_Year_To_Date >= ytd_threshold & 
           prepared_data$yrs_residence >= residence_threshold, 1, 0)
)

# Split the data into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(prepared_data$Eligible, p = 0.8, list = FALSE)
trainData <- prepared_data[trainIndex, ]
testData <- prepared_data[-trainIndex, ]

# Train random forest model with cross-validation on the training set
rf_model <- train(
  Eligible ~ Annual.Salary + Gross_Year_To_Date + yrs_residence,
  data = trainData,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5), # 5-fold cross-validation
  ntree = 10
)

# Predict on the test set
test_predictions <- predict(rf_model, testData)

# Calculate metrics for the test set
conf_matrix <- confusionMatrix(test_predictions, testData$Eligible)
precision <- posPredValue(test_predictions, testData$Eligible, positive = "1")
recall <- sensitivity(test_predictions, testData$Eligible, positive = "1")
f1_score <- (2 * precision * recall) / (precision + recall)

# UI Definition
ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "flatly"),
  
  titlePanel("Satellite Service Eligibility Checker"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      # Input fields
      numericInput("annual_salary", "Annual Salary ($):", 
                   value = NA, min = 0, step = 1000),
      
      numericInput("gross_ytd", "Gross Year to Date ($):", 
                   value = NA, min = 0, step = 1000),
      
      numericInput("years_residence", "Years of Residence:", 
                   value = NA, min = 0, step = 0.1),
      
      actionButton("check_eligibility", "Check Eligibility",
                   class = "btn-primary btn-lg btn-block"),
      
      hr(),
      
      actionButton("reset", "Reset Form",
                   class = "btn-secondary btn-block"),
      
      hr(),
      
      # Display model metrics
      h4("Model Metrics"),
      verbatimTextOutput("class_distribution"),
      verbatimTextOutput("conf_matrix"),
      verbatimTextOutput("precision_recall_f1")
    ),
    
    mainPanel(
      width = 9,
      tabsetPanel(
        tabPanel("Results",
                 br(),
                 fluidRow(
                   column(12,
                          uiOutput("eligibility_result")
                   )
                 ),
                 hr(),
                 fluidRow(
                   column(6,
                          h4("Eligibility Factors"),
                          tableOutput("factor_table")
                   ),
                   column(6,
                          h4("Model Prediction"),
                          verbatimTextOutput("debug_info")
                   )
                 )
        ),
        tabPanel("Historical Data",
                 br(),
                 DTOutput("history_table")
        )
      )
    )
  )
)

# Server Definition
server <- function(input, output, session) {
  
  # Store assessment history
  history <- reactiveVal(data.frame())
  
  # Calculate eligibility
  eligibility_calc <- eventReactive(input$check_eligibility, {
    req(input$annual_salary, input$gross_ytd, input$years_residence)
    
    # Simple threshold checks
    meets_criteria <- list(
      salary = input$annual_salary >= salary_threshold,
      ytd = input$gross_ytd >= ytd_threshold,
      residence = input$years_residence >= residence_threshold
    )
    
    # Overall eligibility based purely on thresholds
    eligible <- meets_criteria$salary && 
      meets_criteria$ytd && 
      meets_criteria$residence
    
    # Update history
    current_history <- history()
    new_record <- data.frame(
      Timestamp = Sys.time(),
      Annual_Salary = input$annual_salary,
      Gross_YTD = input$gross_ytd,
      Years_Residence = input$years_residence,
      Eligible = ifelse(eligible, "Yes", "No")
    )
    history(rbind(current_history, new_record))
    
    list(
      eligible = eligible,
      meets_criteria = meets_criteria
    )
  })
  
  # Render eligibility result
  output$eligibility_result <- renderUI({
    result <- eligibility_calc()
    
    div(
      class = ifelse(result$eligible, "alert alert-success", "alert alert-danger"),
      h3(
        ifelse(result$eligible,
               "Customer is eligible for the satellite service.",
               "Customer is not eligible for the satellite service."),
        style = "text-align: center;"
      )
    )
  })
  
  # Debug information
  output$debug_info <- renderPrint({
    result <- eligibility_calc()
    cat("Eligibility Criteria:\n")
    cat("Annual Salary: ", input$annual_salary, " >= ", salary_threshold, " : ", input$annual_salary >= salary_threshold, "\n")
    cat("Gross YTD: ", input$gross_ytd, " >= ", ytd_threshold, " : ", input$gross_ytd >= ytd_threshold, "\n")
    cat("Years Residence: ", input$years_residence, " >= ", residence_threshold, " : ", input$years_residence >= residence_threshold, "\n")
    cat("\nFinal Eligibility: ", result$eligible)
  })
  
  # Render factors table
  output$factor_table <- renderTable({
    result <- eligibility_calc()
    data.frame(
      Factor = c("Annual Salary", "Gross YTD", "Years of Residence"),
      Status = c(
        ifelse(result$meets_criteria$salary, "✓", "✗"),
        ifelse(result$meets_criteria$ytd, "✓", "✗"),
        ifelse(result$meets_criteria$residence, "✓", "✗")
      ),
      Threshold = c(
        paste0("≥ $", format(salary_threshold, big.mark = ",")),
        paste0("≥ $", format(ytd_threshold, big.mark = ",")),
        paste0("≥ ", format(residence_threshold, digits = 1))
      )
    )
  })
  
  # Display model metrics
  output$class_distribution <- renderText({
    paste("Class Distribution - Eligible: ", sum(prepared_data$Eligible == 1),
          ", Not Eligible: ", sum(prepared_data$Eligible == 0))
  })
  
  # Render confusion matrix
  output$conf_matrix <- renderPrint({
    conf_matrix
  })
  
  # Render precision, recall, and F1-score
  output$precision_recall_f1 <- renderPrint({
    cat("Precision:", round(precision, 4), "\n")
    cat("Recall:", round(recall, 4), "\n")
    cat("F1 Score:", round(f1_score, 4), "\n")
  })
  
  # Render history table
  output$history_table <- renderDT({
    datatable(history(),
              options = list(pageLength = 10,
                             order = list(list(0, 'desc'))),
              rownames = FALSE)
  })
  
  # Reset form
  observeEvent(input$reset, {
    updateNumericInput(session, "annual_salary", value = NA)
    updateNumericInput(session, "gross_ytd", value = NA)
    updateNumericInput(session, "years_residence", value = NA)
  })
}

# Run the application
shinyApp(ui = ui, server = server)
