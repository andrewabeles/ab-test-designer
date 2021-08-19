#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(
    titlePanel("Power Calculator"),
    sidebarLayout(
        sidebarPanel(
            sliderInput("baseline",
                        "Baseline:",
                        min = 0,
                        max = 1,
                        value = 0.5),
            sliderInput("alpha",
                        "Alpha:",
                        min = 0, 
                        max = 1,
                        value = 0.05),
            sliderInput("power",
                        "Power:",
                        min = 0, 
                        max = 1,
                        value = 0.8)
        ),
        mainPanel(
           plotOutput("plot")
        )
    )
)

server <- function(input, output) {
    output$plot <- renderPlot({
        effect_sizes <- seq(0.01, 1, by = 0.01)
        sample_sizes <- c()
        for (e in effect_sizes) {
            p2 <- input$baseline * (1 + e) 
            if (p2 > 1) {
                p2 <- 1
            }
            n <- power.prop.test(
                n = NULL,
                p1 = input$baseline,
                p2 = p2,
                sig.level = input$alpha,
                power = input$power
            )$n
            sample_sizes <- append(sample_sizes, n)
        }
        plot(x = effect_sizes, y = sample_sizes, type = "o")
    })
}

shinyApp(ui = ui, server = server)
