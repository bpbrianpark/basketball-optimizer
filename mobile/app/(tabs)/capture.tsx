import { StyleSheet } from 'react-native';
import { Link } from 'expo-router';
import { Text, View } from '@/components/Themed';

export default function CaptureScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Record or pick video</Text>
      <Text style={styles.subtitle}>
        Camera recording and gallery picker will be added here (MOB-3, MOB-4).
      </Text>
      <View style={styles.separator} lightColor="#eee" darkColor="rgba(255,255,255,0.1)" />
      <Link href="/(tabs)/results" asChild>
        <Text style={styles.link}>View Results (placeholder) →</Text>
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    marginTop: 12,
    textAlign: 'center',
    opacity: 0.8,
  },
  separator: {
    marginVertical: 30,
    height: 1,
    width: '80%',
  },
  link: {
    fontSize: 16,
    color: '#2f95dc',
  },
});
