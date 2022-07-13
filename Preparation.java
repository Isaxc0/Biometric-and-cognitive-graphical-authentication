import javax.swing.*;
import java.awt.*;
import java.io.File;
import javafx.beans.binding.Bindings;
import javafx.beans.property.DoubleProperty;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.media.*;

/**
 * Displays the initial GUI and collects the participants ID
 *
 * @author Isaac Baldwin
 */
public class Preparation {
    private String ID; //participant ID
    private JFrame frame; //main JFrame


    public Preparation() {
        frame = new JFrame();
        menu();
    }

    /**
     * Displays the initial menu to collect the participants ID
     */
    private void menu() {
        frame = new JFrame();
        frame.setTitle("Pattern drawing menu");
        frame.setLocation(100, 100);
        frame.setSize(400, 300);

        JTextField textArea = new JTextField(); // text area to allow participant to enter their ID
        textArea.setBounds(200, 82, 130, 30);

        JButton button = new JButton("Submit"); // submit button to allow participant to submit their ID
        button.setBounds(150, 175, 100, 30);
        button.addActionListener(e -> {
            ID = textArea.getText();
            frame.getContentPane().removeAll();
            instructions(); // move on to the instructional video
        });

        JLabel label = new JLabel();
        label.setText("Participant ID :");
        label.setFont(new Font("Calibri", Font.BOLD, 25));
        label.setBounds(20, 50, 300, 100);

        frame.add(textArea);
        frame.add(label);
        frame.add(button);

        frame.setLayout(null);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }


    /**
     * Displays the instructional video before calling to show the first stage pattern
     */
    private void instructions() {
        frame.setTitle("Instructions");
        frame.setLayout(new BorderLayout());
        frame.setSize(800, 900);
        JButton button = new JButton("Start pattern drawing");
        button.addActionListener(e -> {
            frame.getContentPane().removeAll();
            new DrawingWindow(ID, frame, 1);
        });
        JTextArea textArea = new JTextArea("Please watch the instructional video below");
        textArea.setFont(new Font("Calibri", Font.BOLD, 20));
        JFXPanel fxPanel = new JFXPanel();
        JPanel botPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        botPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        botPanel.add(new JLabel(""));  // add padding to continue button
        botPanel.add(button, BorderLayout.SOUTH);

        File videoSource = new File("src\\video.mp4");
        Media m = new Media(videoSource.toURI().toString());
        MediaPlayer player = new MediaPlayer(m);
        MediaView viewer = new MediaView(player);
        StackPane root = new StackPane();
        Scene scene = new Scene(root);

        // fit the video resolution onto the panel
        DoubleProperty width = viewer.fitWidthProperty();
        DoubleProperty height = viewer.fitHeightProperty();
        width.bind(Bindings.selectDouble(viewer.sceneProperty(), "width"));
        height.bind(Bindings.selectDouble(viewer.sceneProperty(), "height"));
        viewer.setPreserveRatio(true);
        root.getChildren().add(viewer);
        player.play();  // play video
        fxPanel.setScene(scene);
        frame.add(textArea, BorderLayout.NORTH);
        frame.add(botPanel, BorderLayout.SOUTH);
        frame.add(fxPanel, BorderLayout.CENTER);
        frame.setVisible(true);
    }

    public static void main(String args[]) {
        new Preparation();
    }
}