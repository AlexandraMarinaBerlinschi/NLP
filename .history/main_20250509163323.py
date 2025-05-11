from Utils.Enums import RomanianParty
from Utils.UtilityFunctions import preprocess_text
from analysis.TopicModelin import ManifestoTopicModeling
from api.ManifestoApi import ManifestoAPI
import pandas as pd


def main():
	api_key = "147c9bb7aefe93dc971088e6210d8f46"
	api = ManifestoAPI(api_key)

	target_manifestos = [
		"93223_201612",
		"93430_201612",
		"93440_201612",
		"93540_201612",
		"93951_201612"
	]

	try:
		manifesto_data = api.fetch_manifestos(target_manifestos)

		manifesto_corpus = []
		party_names = []

		for manifesto in manifesto_data:
			print(f"\nLoaded manifesto for party: {RomanianParty.get_name(manifesto.metadata.party_id)}")

			full_text = ' '.join(item.text for item in manifesto.text_items)
			full_text = preprocess_text(full_text.strip())

			manifesto_corpus.append(full_text)
			party_names.append(RomanianParty.get_name(manifesto.metadata.party_id))

	except Exception as e:
		print(f"Error occurred: {str(e)}")
		import traceback
		print(f"Full traceback: {traceback.format_exc()}")
		return

	# ===== TOPIC MODELING =====
	topic_modeler = ManifestoTopicModeling()
	model, party_topics, topic_df = topic_modeler.perform_topic_modeling(
		manifesto_corpus,
		party_names
	)

	# afisam rezultatele
	print("\nDetailed Topic Analysis:")
	pd.set_option('display.max_colwidth', None)
	print(topic_df)


if __name__ == "__main__":
	main()
