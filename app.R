library(shiny)
library(ggplot2)
library(plotly)
library(rhandsontable)

ui <- navbarPage("A/B Test Designer",
        tabPanel("Proportions",
            sidebarLayout(
                sidebarPanel(
                  textInput("metric_name_proportion", "Metric Name: "),
                    numericInput("users_per_week_proportion",
                                 "Users per Week",
                                 value = 10000),
                    sliderInput("traffic_pct_proportion",
                                "Percent of Traffic",
                                min = 0,
                                max = 1,
                                value = 1),
                    sliderInput("baseline_proportion",
                                "Baseline",
                                min = 0.01,
                                max = 0.99,
                                value = 0.5),
                    selectInput("alpha_proportion",
                                "Alpha",
                                choices = c(0.01, 0.05, 0.1),
                                selected = 0.05),
                    selectInput("power_proportion",
                                "Power",
                                choices = c(0.8, 0.9, 0.99),
                                selected = 0.8),
                    actionButton("save_proportion","Save")
                ),
                mainPanel(
                    plotlyOutput("plot_proportions", height=500),
                    sliderInput("effect_size_range_proportion",
                                "Lift Range",
                                width = "100%",
                                min = 0.01,
                                max = 1,
                                value = c(0.05, 0.25))
                )
            )
        ),
        tabPanel("Means",
            sidebarLayout(
                sidebarPanel(
                    textInput("metric_name_mean", "Metric Name: "),
                    numericInput("users_per_week_mean",
                                 "Users per Week",
                                 value = 10000),
                    sliderInput("traffic_pct_mean",
                                "Percent of Traffic",
                                min = 0,
                                max = 1,
                                value = 1),
                    numericInput("baseline_mean",
                                 "Baseline",
                                 value = 3.14),
                    numericInput("sd",
                                 "Standard Deviation",
                                 value = 2.71),
                    selectInput("alpha_mean",
                                "Alpha",
                                choices = c(0.01, 0.05, 0.1),
                                selected = 0.05),
                    selectInput("power_mean",
                                "Power",
                                choices = c(0.8, 0.9, 0.99),
                                selected = 0.8),
                    actionButton("save_mean","Save")
                ),
                mainPanel(
                    plotlyOutput("plot_means", height=550),
                    sliderInput("effect_size_range_mean",
                                "Lift Range",
                                width = "100%",
                                min = 0.01,
                                max = 1,
                                value = c(0.05, 0.25))
                )
            )
        ),
        tabPanel("History", 
          mainPanel(width=12, rHandsontableOutput('history_table'))
        ),
        tabPanel("About",
                 p("This is an app for estimating the sample size and amount of time needed to run an A/B test. 
                   It takes the following parameters as input: users per week, percent of traffic, baseline, standard deviation (only for means), alpha, and power.
                   It outputs the total sample size and number of weeks required to detect with statistical significance a given lift over the baseline. 
                   If you find an experiment design that you like, you can give it a name and click 'Save' to record it in the History tab. You can then right click 
                   the history table to export it as a CSV."),
                 h3("Users per Week"),
                 p("The number of users per week that will be included in the experiment."),
                 h3("Percent of Traffic"),
                 p("The percent of traffic that will be included in the experiment. This is used to calculate the actual number of users per week that will be included."),
                 h3("Baseline"),
                 p("The baseline value of the variable being tested. The variable's proportion or average in the control group."),
                 h3("Standard Deviation"),
                 p("The standard deviation of the baseline variable. 
                   Larger standard deviations require larger sample sizes to ensure the observed difference between groups is not due to the baseline variable's natural variance."),
                 h3("Lift"),
                 p("The treatment's lift over the baseline. The percentage difference between the treatment and control."),
                 h3("Alpha"),
                 p("Also known as the significance level, alpha is the experiment's Type I error (false positive) rate."),
                 h3("Power"),
                 p("1 - Beta, the Type II error or false negative rate. Power represents the probability the experiment will detect a given percent lift at a given significance level.")
        )
    )

cal_proportion <- function(input) {
  p1 <- input$baseline_proportion
  effect_size_range <- input$effect_size_range_proportion
  effect_sizes <- seq(effect_size_range[1], effect_size_range[2], by = 0.01)
  sample_sizes <- c()
  for (e in effect_sizes) {
    p2 <- p1 * (1 + e)
    if (p2 > 1) {p2 <- 1}
    n <- power.prop.test(
      n = NULL,
      p1 = p1,
      p2 = p2,
      sig.level = as.numeric(input$alpha_proportion),
      power = as.numeric(input$power_proportion)
    )$n * 2 # the function outputs required sample size per group, so we double to get the total required sample size 
    sample_sizes <- append(sample_sizes, ceiling(n))
  }
  users_per_week <- input$users_per_week_proportion * input$traffic_pct_proportion 
  weeks <- signif(sample_sizes/users_per_week, 2)
  data <- data.frame(effect_sizes, sample_sizes, weeks)
  colnames(data) <- c("Lift", "SampleSize", "Weeks")
  return (data)
}

cal_mean <- function(input) {
  x1 <- input$baseline_mean
  effect_size_range <- input$effect_size_range_mean
  effect_sizes <- seq(effect_size_range[1], effect_size_range[2], by = 0.01)
  sample_sizes <- c()
  for (e in effect_sizes) {
    x2 <- x1 * (1 + e)
    n <- power.t.test(
      n = NULL, 
      delta = x2 - x1,
      sd = input$sd,
      sig.level = as.numeric(input$alpha_mean),
      power = as.numeric(input$power_mean)
    )$n * 2 # the function outputs required sample size per group, so we double to get the total required sample size 
    sample_sizes <- append(sample_sizes, ceiling(n))
  }
  users_per_week <- input$users_per_week_mean * input$traffic_pct_mean 
  weeks <- signif(sample_sizes/users_per_week, 2)
  data <- data.frame(effect_sizes, sample_sizes, weeks)
  colnames(data) <- c("Lift", "SampleSize", "Weeks")
  return (data)
}

get_proportion_input <- function(input) {
  input_values <- c(input$metric_name_proportion
                   ,input$users_per_week_proportion
                   ,input$traffic_pct_proportion
                   ,input$baseline_proportion
                   ,"N/A"
                   ,input$alpha_proportion
                   ,input$power_proportion)
  return (input_values)
}

get_mean_input <- function(input) {
  input_values <- c(input$metric_name_mean
                    ,input$users_per_week_mean
                    ,input$traffic_pct_mean
                    ,input$baseline_mean
                    ,input$sd
                    ,input$alpha_mean
                    ,input$power_mean)
  return (input_values)
}

make_history_row <- function (input_data, output_data) {
  # this function creates the table, 
  # insert data as first row into the table
  
  inputs <- paste(input_data, " ", sep="")
  display_effective_size <- seq(0.05, 1, by = 0.05)
  max_display_index <- length(display_effective_size)
  transformed_output <- as.data.frame(t(output_data))
  effect_size <- c()
  weeks <- c()
  current_display_index <- 1
  for (d in transformed_output) {
    # effective_size = d[1], weeks = d[3]
    while (current_display_index <= max_display_index && round(d[1], digits = 2) > round(display_effective_size[current_display_index], digits = 2)){
        # make display effective size align with actual data
        effect_size <- append(effect_size, paste("Lift=", display_effective_size[current_display_index], sep=""))
        weeks <- append(weeks, "N/A")
        current_display_index <- current_display_index + 1
    }
    if (current_display_index <= max_display_index && round(d[1], digits = 2) == round(display_effective_size[current_display_index], digits = 2)) {
      effect_size <- append(effect_size, paste("Lift=", display_effective_size[current_display_index], sep=""))
      weeks <- append(weeks, d[3])  
      current_display_index <- current_display_index + 1
    } else if (current_display_index > max_display_index) {
      break
    } 
  }
  if (current_display_index <=  max_display_index) {
    while (current_display_index <=  max_display_index) {
      effect_size <- append(effect_size, paste("Lift=", display_effective_size[current_display_index], sep=""))
      weeks <- append(weeks, "N/A")
      current_display_index <- current_display_index + 1
    }
  }

  df <- data.frame( matrix(inputs, nrow = 1), matrix(weeks, nrow=1))
  names(df) <- c("Metric Name", "Users per Week", "Percent of Traffic", "Baseline", "SD", "Alpha", "Power", effect_size)
  
  return (df)
} 

render_history_table <- function (table_content) {
  renderRHandsontable({
    rhandsontable(table_content, width="100%", fixedColumnsLeft=7, useTypes = FALSE) %>%
      hot_context_menu(
        customOpts = list(
          csv = list(name = "Download to CSV",
                     callback = htmlwidgets::JS(
                       "function (key, options) {
                         var csv = csvString(this);

                         var link = document.createElement('a');
                         link.setAttribute('href', 'data:text/plain;charset=utf-8,' +
                           encodeURIComponent(csv));
                         link.setAttribute('download', 'sample_size.csv');

                         document.body.appendChild(link);
                         link.click();
                         document.body.removeChild(link);
                       }"))))
    })
}

server <- function(input, output) {
    output$plot_proportions <- renderPlotly({
      data <- cal_proportion(input)
      ggplot(data, aes(x = Lift, y = Weeks, text = paste("SampleSize:", SampleSize))) + 
        geom_line(color = "grey") + 
        geom_point(color = "coral") +
        xlab("Lift") + 
        ylab("Weeks") + 
        scale_x_continuous(labels = scales::percent)
    })
    output$plot_means <- renderPlotly({
      data <- cal_mean (input)
      ggplot(data, aes(x = Lift, y = Weeks, text = paste("SampleSize:", SampleSize))) +
        geom_line(color = "grey") +
        geom_point(color = "coral") + 
        xlab("Lift") + 
        ylab("Weeks") + 
        scale_x_continuous(labels = scales::percent)
    })
    
    values <- reactiveValues()
    first_row <- 1
    
    history_proportion <- eventReactive(input$save_proportion, {
      output_data <- cal_proportion(input)
      input_data <- get_proportion_input (input)
      if (first_row == 1 && input$save_proportion == 1){
        first_row <<- 0
        values$df <- make_history_row(input_data, output_data)
      } else {
        df_new <- make_history_row(input_data, output_data)
        values$df <- rbind(df_new, values$df)
      }
      return (values$df)
    })
    
    history_mean <- eventReactive(input$save_mean, {
      output_data <- cal_mean(input)
      input_data <- get_mean_input (input)
      if (first_row == 1 && input$save_mean == 1){
        first_row <<- 0
        values$df <- make_history_row(input_data, output_data)
      } else {
        df_new <- make_history_row(input_data, output_data)
        values$df <- rbind(df_new, values$df)
      }
      return (values$df)
    })
    
    observeEvent(history_proportion(), {
      output$history_table <- render_history_table(history_proportion())
    })
    
    observeEvent(history_mean(), {
      output$history_table <- render_history_table(history_mean())
    })
}

shinyApp(ui = ui, server = server)
