Add-Type -AssemblyName PresentationFramework

$window = New-Object System.Windows.Window
$window.WindowStartupLocation = "CenterScreen"
$window.Width = 800
$window.Height = 400
$window.Topmost = $true
$window.Background = "LimeGreen"
$window.WindowStyle = "None"

$text = New-Object System.Windows.Controls.TextBlock
$text.Text = "✅`nTRAINING TERMINÉ !"
$text.FontSize = 60
$text.FontWeight = "Bold"
$text.Foreground = "Black"
$text.HorizontalAlignment = "Center"
$text.VerticalAlignment = "Center"
$text.TextAlignment = "Center"

$window.Content = $text
$window.Add_MouseDown({ $window.Close() })

$window.ShowDialog()